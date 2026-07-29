from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import geopandas as gpd
from spatialdata import SpatialData
from spatialdata._io.io_shapes import _read_shapes as _spatialdata_read_shapes
from spatialdata.models import ShapesModel

from napari_harpy.core.validation import normalize_spatialdata_name

_ELEMENT_CONTAINERS = ("images", "labels", "points", "shapes", "tables")
_RECOVERY_ROOT_NAME = ".harpy_recovery"

ShapesElementValidator = Callable[[gpd.GeoDataFrame], None]


def load_shapes_element_from_store(
    sdata: SpatialData,
    shapes_name: str,
    *,
    validate_before_install: ShapesElementValidator | None = None,
) -> gpd.GeoDataFrame:
    """Load one exact persisted Shapes element into the live SpatialData."""
    normalized_name = normalize_spatialdata_name(shapes_name, "Shapes element name")
    _require_backed_path(sdata)
    disk_types = _element_types_on_disk(sdata, normalized_name)
    _raise_if_name_used_by_non_shapes_element(
        disk_types,
        shapes_name=normalized_name,
        location="backing store",
    )
    if "shapes" not in disk_types:
        raise ValueError(f"Shapes element `{normalized_name}` does not exist in the backing store.")
    _raise_if_name_used_by_non_shapes_element(
        _element_types_in_memory(sdata, normalized_name),
        shapes_name=normalized_name,
        location="live SpatialData",
    )

    candidate = _read_shapes_element_from_store(sdata, normalized_name)
    ShapesModel.validate(candidate)
    if validate_before_install is not None:
        if not callable(validate_before_install):
            raise TypeError("`validate_before_install` must be callable or None.")
        validate_before_install(candidate)

    previous = sdata.shapes.get(normalized_name)
    try:
        sdata.shapes[normalized_name] = candidate
    except Exception:
        _restore_live_shapes_value(sdata, normalized_name, previous)
        raise
    return sdata.shapes[normalized_name]


def write_shapes_element(
    sdata: SpatialData,
    shapes_name: str,
    element: gpd.GeoDataFrame,
    *,
    overwrite: bool,
) -> gpd.GeoDataFrame:
    """Install one Shapes element and persist it when SpatialData is backed.

    An existing backed element is replaced through this recovery-aware
    lifecycle (using ``tumor`` as an example)::

        shapes/tumor                           # current element on disk
        shapes/tumor__napari_harpy_stage_...  # new element, fully serialized on disk under a staging name
            ↓
        .harpy_recovery/shapes__tumor__...     # move current element here
            ↓
        shapes/tumor                           # move staged replacement here
            ↓
        read, validate, and consolidate metadata
            ├── failure → restore recovery copy to shapes/tumor
            └── success → delete recovery copy

    The previous element therefore remains recoverable until the replacement
    has been read successfully and its store metadata has been consolidated.
    """
    normalized_name = normalize_spatialdata_name(shapes_name, "Shapes element name")
    ShapesModel.validate(element)

    live_types = _element_types_in_memory(sdata, normalized_name)
    _raise_if_name_used_by_non_shapes_element(
        live_types,
        shapes_name=normalized_name,
        location="live SpatialData",
    )
    shapes_exists_in_memory = "shapes" in live_types

    if not sdata.is_backed():
        if shapes_exists_in_memory and not overwrite:
            raise ValueError(f"Shapes element `{normalized_name}` already exists. Set `overwrite=True` to replace it.")
        previous = sdata.shapes.get(normalized_name)
        try:
            sdata.shapes[normalized_name] = element
        except Exception:
            _restore_live_shapes_value(sdata, normalized_name, previous)
            raise
        return sdata.shapes[normalized_name]

    store_path = _require_backed_path(sdata)
    disk_types = _element_types_on_disk(sdata, normalized_name)
    _raise_if_name_used_by_non_shapes_element(
        disk_types,
        shapes_name=normalized_name,
        location="backing store",
    )
    shapes_exists_on_disk = "shapes" in disk_types
    if shapes_exists_in_memory != shapes_exists_on_disk:
        raise ValueError(
            f"Shapes element `{normalized_name}` must have matching live and persisted presence before it can be "
            "written."
        )
    if shapes_exists_in_memory and not overwrite:
        raise ValueError(f"Shapes element `{normalized_name}` already exists. Set `overwrite=True` to replace it.")

    previous_live_element = sdata.shapes.get(normalized_name)
    shapes_path = store_path / "shapes"
    shapes_collection_existed_before_write = shapes_path.exists()
    requested_path = shapes_path / normalized_name
    staging_name = f"{normalized_name}__napari_harpy_stage_{uuid4().hex}"
    staging_path = shapes_path / staging_name
    recovery_root = store_path / _RECOVERY_ROOT_NAME
    previous_element_recovery_path: Path | None = None

    # Persist a complete staging Shapes element before modifying the requested
    # persisted Shapes element. A staging failure leaves it unchanged.
    try:
        _write_staging_shapes_element(
            sdata,
            store_path=store_path,
            staging_name=staging_name,
            element=element,
        )
    except Exception as error:
        cleanup_path = staging_path if shapes_collection_existed_before_write else shapes_path
        try:
            _remove_directory_if_present(cleanup_path)
        except Exception as cleanup_error:  # noqa: BLE001
            _raise_shapes_write_recovery_error(
                shapes_name=normalized_name,
                original_error=error,
                recovery_errors=[f"could not remove the incomplete staging Shapes element: {cleanup_error}"],
                possibly_retained_paths=[cleanup_path],
            )
        raise

    # Rename the already serialized staging directory into place. If any
    # commit step fails, restore the previous directory and live Shapes value.
    previous_element_moved_to_recovery = False
    new_element_moved_to_requested_path = False
    try:
        if shapes_exists_on_disk:
            _prepare_recovery_root(recovery_root)
            previous_element_recovery_path = recovery_root / f"shapes__{normalized_name}__{uuid4().hex}"
            os.replace(requested_path, previous_element_recovery_path)
            previous_element_moved_to_recovery = True
        os.replace(staging_path, requested_path)
        new_element_moved_to_requested_path = True
        # Read back what was actually persisted so the in-memory and on-disk
        # representations match. SpatialData validates the materialized value
        # through Shapes.__setitem__ during the assignment below.
        committed_element = _read_shapes_element_from_store(sdata, normalized_name)
        sdata.shapes[normalized_name] = committed_element
    except Exception as error:
        rollback_errors = _restore_shapes_after_failed_swap(
            sdata,
            shapes_name=normalized_name,
            requested_path=requested_path,
            staging_path=staging_path,
            recovery_root=recovery_root,
            previous_element_recovery_path=previous_element_recovery_path,
            previous_element_moved_to_recovery=previous_element_moved_to_recovery,
            new_element_moved_to_requested_path=new_element_moved_to_requested_path,
            shapes_path=shapes_path,
            shapes_collection_existed_before_write=shapes_collection_existed_before_write,
            previous_live_element=previous_live_element,
        )
        if rollback_errors:
            _raise_shapes_write_recovery_error(
                shapes_name=normalized_name,
                original_error=error,
                recovery_errors=rollback_errors,
                possibly_retained_paths=[
                    requested_path,
                    staging_path,
                    *(
                        [previous_element_recovery_path]
                        if previous_element_recovery_path is not None
                        else []
                    ),
                    *([shapes_path] if not shapes_collection_existed_before_write else []),
                ],
            )
        raise

    try:
        sdata.write_consolidated_metadata()
    except Exception as error:
        rollback_errors = _restore_shapes_after_failed_swap(
            sdata,
            shapes_name=normalized_name,
            requested_path=requested_path,
            staging_path=staging_path,
            recovery_root=recovery_root,
            previous_element_recovery_path=previous_element_recovery_path,
            previous_element_moved_to_recovery=previous_element_moved_to_recovery,
            new_element_moved_to_requested_path=new_element_moved_to_requested_path,
            shapes_path=shapes_path,
            shapes_collection_existed_before_write=shapes_collection_existed_before_write,
            previous_live_element=previous_live_element,
        )
        if not rollback_errors:
            try:
                sdata.write_consolidated_metadata()
            except Exception as recovery_error:  # noqa: BLE001
                rollback_errors.append(f"could not consolidate the restored SpatialData state: {recovery_error}")
        if rollback_errors:
            _raise_shapes_write_recovery_error(
                shapes_name=normalized_name,
                original_error=error,
                recovery_errors=rollback_errors,
                possibly_retained_paths=[
                    requested_path,
                    staging_path,
                    *(
                        [previous_element_recovery_path]
                        if previous_element_recovery_path is not None
                        else []
                    ),
                    *([shapes_path] if not shapes_collection_existed_before_write else []),
                ],
            )
        raise

    if previous_element_recovery_path is not None:
        try:
            _remove_directory_if_present(previous_element_recovery_path)
        except Exception as error:
            raise RuntimeError(
                f"Shapes element `{normalized_name}` was committed at `{requested_path}`, but write finalization "
                "failed because its previous recovery copy could not be removed. The consolidated store remains "
                f"valid. Cleanup error: {error}. Retained recovery path: `{previous_element_recovery_path}`."
            ) from error
        _remove_empty_recovery_root(recovery_root)
    return sdata.shapes[normalized_name]


def _read_shapes_element_from_store(
    sdata: SpatialData,
    shapes_name: str,
) -> gpd.GeoDataFrame:
    """Read one exact Shapes payload without changing live SpatialData."""
    path = _require_backed_path(sdata) / "shapes" / shapes_name
    return _spatialdata_read_shapes(path)


def _require_backed_path(sdata: SpatialData) -> Path:
    if not sdata.is_backed() or sdata.path is None:
        raise ValueError("SpatialData must be backed by a zarr store.")
    path = Path(sdata.path)
    if not path.is_dir():
        raise ValueError("SpatialData must be backed by a local directory zarr store.")
    return path


def _element_types_in_memory(sdata: SpatialData, element_name: str) -> set[str]:
    return {
        container_name
        for container_name in _ELEMENT_CONTAINERS
        if element_name in getattr(sdata, container_name)
    }


def _element_types_on_disk(sdata: SpatialData, element_name: str) -> set[str]:
    element_types: set[str] = set()
    for element_path in sdata.elements_paths_on_disk():
        element_type, separator, current_name = element_path.partition("/")
        if separator and current_name == element_name:
            element_types.add(element_type)
    return element_types


def _raise_if_name_used_by_non_shapes_element(
    element_types: set[str],
    *,
    shapes_name: str,
    location: str,
) -> None:
    conflicting_types = element_types - {"shapes"}
    if conflicting_types:
        formatted_types = ", ".join(sorted(conflicting_types))
        raise ValueError(
            f"Element `{shapes_name}` exists as {formatted_types} in the {location}; it cannot be used as a Shapes "
            "element."
        )


def _write_staging_shapes_element(
    sdata: SpatialData,
    *,
    store_path: Path,
    staging_name: str,
    element: gpd.GeoDataFrame,
) -> None:
    """Serialize one staging Shapes element without consolidating metadata."""
    # Keep this private SpatialData compatibility boundary isolated here. The
    # public write_element() would consolidate metadata after this temporary
    # write, while _write_element() retains SpatialData's Shapes serializer.
    sdata._write_element(
        element=element,
        zarr_container_path=store_path,
        element_type="shapes",
        element_name=staging_name,
        overwrite=False,
    )


def _restore_shapes_after_failed_swap(
    sdata: SpatialData,
    *,
    shapes_name: str,
    requested_path: Path,
    staging_path: Path,
    recovery_root: Path,
    previous_element_recovery_path: Path | None,
    previous_element_moved_to_recovery: bool,
    new_element_moved_to_requested_path: bool,
    shapes_path: Path,
    shapes_collection_existed_before_write: bool,
    previous_live_element: gpd.GeoDataFrame | None,
) -> list[str]:
    """Restore the previous accepted Shapes state after a staged write fails to commit.

    Each state below is ordered as
    ``previous_element_moved_to_recovery`` and
    ``new_element_moved_to_requested_path``.

    Recovery states
    ---------------
    ``False, False``
        No requested-path rename completed. Remove the staged element, or
        remove the complete Shapes collection if it was created by this
        rejected write.
    ``True, False``
        The previous element moved to recovery, but the staged-element rename
        failed. Restore the previous element.
    ``False, True``
        A newly created element reached the requested path. Remove the
        rejected new element, or the newly created Shapes collection.
    ``True, True``
        The replacement reached the requested path and the previous element
        remains recoverable. Remove the rejected replacement, then restore
        the previous element.
    """
    errors: list[str] = []
    if not shapes_collection_existed_before_write:
        try:
            _remove_directory_if_present(shapes_path)
        except Exception as error:  # noqa: BLE001
            errors.append(f"could not remove the Shapes collection created by the rejected write: {error}")
    else:
        if new_element_moved_to_requested_path:
            try:
                _remove_directory_if_present(requested_path)
            except Exception as error:  # noqa: BLE001
                errors.append(f"could not remove the rejected persisted Shapes element: {error}")

        if previous_element_moved_to_recovery:
            if previous_element_recovery_path is None:  # pragma: no cover - construction invariant
                errors.append(
                    "could not restore the previous persisted Shapes element because its recovery path is absent"
                )
            elif requested_path.exists():
                errors.append(
                    "could not restore the previous persisted Shapes element because the rejected element remains"
                )
            else:
                try:
                    os.replace(previous_element_recovery_path, requested_path)
                except Exception as error:  # noqa: BLE001
                    errors.append(f"could not restore the previous persisted Shapes element: {error}")

        try:
            _remove_directory_if_present(staging_path)
        except Exception as error:  # noqa: BLE001
            errors.append(f"could not remove the staging Shapes element: {error}")

    try:
        _restore_live_shapes_value(sdata, shapes_name, previous_live_element)
    except Exception as error:  # noqa: BLE001
        errors.append(f"could not restore the previous live Shapes element: {error}")
    _remove_empty_recovery_root(recovery_root)
    return errors


def _prepare_recovery_root(recovery_root: Path) -> None:
    """Create or validate the plain filesystem directory used for recovery."""
    if recovery_root.is_symlink():
        raise ValueError(f"Shapes recovery path `{recovery_root}` must not be a symbolic link.")
    if recovery_root.exists():
        if not recovery_root.is_dir():
            raise ValueError(f"Shapes recovery path `{recovery_root}` must be a directory.")
        if (recovery_root / "zarr.json").exists():
            raise ValueError(f"Shapes recovery path `{recovery_root}` must not be a Zarr group.")
        return
    recovery_root.mkdir()


def _remove_empty_recovery_root(recovery_root: Path) -> None:
    try:
        recovery_root.rmdir()
    except OSError:
        pass


def _remove_directory_if_present(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _restore_live_shapes_value(
    sdata: SpatialData,
    shapes_name: str,
    previous: gpd.GeoDataFrame | None,
) -> None:
    if previous is None:
        sdata.shapes.pop(shapes_name, None)
    else:
        sdata.shapes[shapes_name] = previous


def _raise_shapes_write_recovery_error(
    *,
    shapes_name: str,
    original_error: Exception,
    recovery_errors: list[str],
    possibly_retained_paths: list[Path],
) -> None:
    details = "; ".join(recovery_errors)
    retained_paths = [path for path in possibly_retained_paths if path.exists()]
    retained_message = (
        " Retained paths: " + ", ".join(f"`{path}`" for path in retained_paths) + "."
        if retained_paths
        else ""
    )
    raise RuntimeError(
        f"Writing Shapes element `{shapes_name}` failed and recovery could not complete. Original error: "
        f"{original_error}. Recovery error: {details}.{retained_message}"
    ) from original_error
