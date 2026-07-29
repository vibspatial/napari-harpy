from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import geopandas as gpd
from spatialdata import SpatialData
from spatialdata._io.io_shapes import _read_shapes as _spatialdata_read_shapes
from spatialdata.models import ShapesModel

from napari_harpy.core.validation import normalize_spatialdata_name

_ELEMENT_CONTAINERS = ("images", "labels", "points", "shapes", "tables")

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
    """Install one Shapes element and persist it when SpatialData is backed."""
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

    _require_backed_path(sdata)
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
    previous_persisted_element = (
        _read_shapes_element_from_store(sdata, normalized_name) if shapes_exists_on_disk else None
    )
    staging_name = f"{normalized_name}__napari_harpy_stage_{uuid4().hex}"

    # Persist a complete staging Shapes element before modifying the requested
    # persisted Shapes element. A staging failure leaves it unchanged.
    try:
        sdata.shapes[staging_name] = element
        try:
            sdata.write_element(staging_name)
        finally:
            sdata.shapes.pop(staging_name, None)
    except Exception as error:
        cleanup_errors = _cleanup_staging_element(sdata, staging_name)
        if cleanup_errors:
            _raise_recovery_error(
                shapes_name=normalized_name,
                staging_name=staging_name,
                original_error=error,
                recovery_errors=cleanup_errors,
            )
        raise

    # Replace the requested persisted Shapes element only after staging
    # succeeds. If this delete-and-rewrite fails, restore the previous
    # persisted and live Shapes elements.
    try:
        if shapes_exists_on_disk:
            sdata.delete_element_from_disk(normalized_name)
        sdata.shapes[normalized_name] = element
        sdata.write_element(normalized_name)
        # Read back what was actually persisted so the in-memory and on-disk
        # representations match. SpatialData validates the materialized value
        # through Shapes.__setitem__ during the assignment below.
        committed_element = _read_shapes_element_from_store(sdata, normalized_name)
        sdata.shapes[normalized_name] = committed_element
    except Exception as error:
        rollback_errors = _restore_previous_backed_shapes_element(
            sdata,
            normalized_name,
            previous_live_element=previous_live_element,
            previous_persisted_element=previous_persisted_element,
        )
        if rollback_errors:
            _raise_recovery_error(
                shapes_name=normalized_name,
                staging_name=staging_name,
                original_error=error,
                recovery_errors=rollback_errors,
            )
        cleanup_errors = _cleanup_staging_element(sdata, staging_name)
        if cleanup_errors:
            _raise_recovery_error(
                shapes_name=normalized_name,
                staging_name=staging_name,
                original_error=error,
                recovery_errors=cleanup_errors,
            )
        raise

    cleanup_errors = _cleanup_staging_element(sdata, staging_name)
    if cleanup_errors:
        details = "; ".join(cleanup_errors)
        raise RuntimeError(
            f"Shapes element `{normalized_name}` was written successfully, but temporary Shapes element "
            f"`{staging_name}` could not be removed. The requested persisted Shapes element remains valid. Cleanup "
            f"error: {details}"
        )
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
    return Path(sdata.path)


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


def _restore_previous_backed_shapes_element(
    sdata: SpatialData,
    shapes_name: str,
    *,
    previous_live_element: gpd.GeoDataFrame | None,
    previous_persisted_element: gpd.GeoDataFrame | None,
) -> list[str]:
    errors: list[str] = []
    try:
        _delete_shapes_from_store_if_present(sdata, shapes_name)
    except Exception as error:  # noqa: BLE001
        errors.append(f"could not remove the partially written Shapes element: {error}")

    if not errors and previous_persisted_element is not None:
        try:
            sdata.shapes[shapes_name] = previous_persisted_element
            sdata.write_element(shapes_name)
        except Exception as error:  # noqa: BLE001
            errors.append(f"could not restore the previous persisted Shapes element: {error}")

    try:
        _restore_live_shapes_value(sdata, shapes_name, previous_live_element)
    except Exception as error:  # noqa: BLE001
        errors.append(f"could not restore the previous live Shapes element: {error}")
    return errors


def _cleanup_staging_element(sdata: SpatialData, staging_name: str) -> list[str]:
    errors: list[str] = []
    sdata.shapes.pop(staging_name, None)
    try:
        _delete_shapes_from_store_if_present(sdata, staging_name)
    except Exception as error:  # noqa: BLE001
        errors.append(str(error))
    return errors


def _delete_shapes_from_store_if_present(sdata: SpatialData, shapes_name: str) -> None:
    disk_types = _element_types_on_disk(sdata, shapes_name)
    _raise_if_name_used_by_non_shapes_element(
        disk_types,
        shapes_name=shapes_name,
        location="backing store",
    )
    if "shapes" in disk_types:
        sdata.delete_element_from_disk(shapes_name)


def _restore_live_shapes_value(
    sdata: SpatialData,
    shapes_name: str,
    previous: gpd.GeoDataFrame | None,
) -> None:
    if previous is None:
        sdata.shapes.pop(shapes_name, None)
    else:
        sdata.shapes[shapes_name] = previous


def _raise_recovery_error(
    *,
    shapes_name: str,
    staging_name: str,
    original_error: Exception,
    recovery_errors: list[str],
) -> None:
    details = "; ".join(recovery_errors)
    raise RuntimeError(
        f"Writing Shapes element `{shapes_name}` failed and recovery could not complete. Temporary Shapes element "
        f"`{staging_name}` was retained when present. Original error: {original_error}. Recovery error: {details}"
    ) from original_error
