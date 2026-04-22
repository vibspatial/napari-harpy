from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from napari.layers import Labels
from spatialdata import get_element_annotators, join_spatialelement_table
from spatialdata.models import TableModel, get_axes_names
from spatialdata.transformations import get_transformation
from xarray import DataArray

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


@dataclass(frozen=True)
class SpatialDataLabelsOption:
    """A selectable labels element discovered from a viewer-linked SpatialData object."""

    label_name: str
    # User-facing text for the dropdown. This may include the dataset name to
    # disambiguate equal label names coming from different SpatialData objects.
    display_name: str
    sdata: SpatialData
    # Selectable coordinate systems for this labels element in the current
    # discovery context.
    coordinate_systems: tuple[str, ...]

    @property
    def identity(self) -> tuple[int, str]:
        """Return a stable identity for preserving widget selection across refreshes."""
        return (id(self.sdata), self.label_name)


@dataclass(frozen=True)
class SpatialDataImageOption:
    """A selectable image element discovered from a viewer-linked SpatialData object."""

    image_name: str
    # User-facing text for the dropdown. This may include the dataset name to
    # disambiguate equal image names coming from different SpatialData objects.
    display_name: str
    sdata: SpatialData
    # Selectable coordinate systems for this image in the current discovery
    # context. When filtered by labels, these are the shared coordinate systems
    # between the selected labels and image elements.
    coordinate_systems: tuple[str, ...]

    @property
    def identity(self) -> tuple[int, str]:
        """Return a stable identity for preserving widget selection across refreshes."""
        return (id(self.sdata), self.image_name)


@dataclass(frozen=True)
class SpatialDataTableMetadata:
    """Metadata that links a table to the labels elements it annotates."""

    table_name: str
    region_key: str
    instance_key: str
    regions: tuple[str, ...]

    def annotates(self, label_name: str) -> bool:
        """Return whether this table can annotate the given labels element."""
        return label_name in self.regions


def get_annotating_table_names(sdata: SpatialData, label_name: str) -> list[str]:
    """Return the table names that annotate a labels element in a SpatialData object."""
    return sorted(get_element_annotators(sdata, label_name))


def get_table(sdata: SpatialData, table_name: str) -> AnnData:
    """Return a validated annotating table."""
    table = sdata[table_name]
    normalize_table_metadata(table)
    return TableModel.validate(table)


def normalize_table_metadata(table: AnnData) -> AnnData:
    """Normalize known SpatialData table attrs into validator-friendly types."""
    _normalize_table_model_attrs(table)
    return table


def get_table_metadata(sdata: SpatialData, table_name: str) -> SpatialDataTableMetadata:
    """Return linkage metadata for an annotating table."""
    table = get_table(sdata, table_name)
    attrs = _get_table_model_attrs(table, table_name)

    return SpatialDataTableMetadata(
        table_name=table_name,
        region_key=str(attrs[TableModel.REGION_KEY_KEY]),
        instance_key=str(attrs[TableModel.INSTANCE_KEY]),
        regions=_normalize_regions(attrs.get(TableModel.REGION_KEY)),
    )


def validate_table_binding(sdata: SpatialData, label_name: str, table_name: str) -> SpatialDataTableMetadata:
    """Validate that a table can be safely bound to a selected labels element."""
    table = get_table(sdata, table_name)
    table_metadata = get_table_metadata(sdata, table_name)

    if not table_metadata.annotates(label_name):
        raise ValueError(f"Table `{table_name}` does not annotate segmentation `{label_name}`.")

    if table_metadata.region_key not in table.obs.columns:
        raise ValueError(f"Table `{table_name}` is missing required obs column `{table_metadata.region_key}`.")

    if table_metadata.instance_key not in table.obs.columns:
        raise ValueError(f"Table `{table_name}` is missing required obs column `{table_metadata.instance_key}`.")

    region_rows = table.obs.loc[table.obs[table_metadata.region_key] == label_name]
    if region_rows.empty:
        return table_metadata

    region_instances = region_rows[table_metadata.instance_key]
    duplicate_instances = region_instances[region_instances.duplicated(keep=False)]
    if not duplicate_instances.empty:
        duplicate_labels = duplicate_instances.astype("string").drop_duplicates().tolist()
        preview = ", ".join(duplicate_labels[:5])
        if len(duplicate_labels) > 5:
            preview += ", ..."
        raise ValueError(
            f"Table `{table_name}` cannot be bound to segmentation `{label_name}` because `{table_metadata.instance_key}` "
            f"contains duplicate values within that region: {preview}."
        )

    return table_metadata


def get_table_obsm_keys(sdata: SpatialData, table_name: str) -> list[str]:
    """Return the available feature matrix keys from `adata.obsm` for a table in a SpatialData object."""
    table = get_table(sdata, table_name)
    return sorted(table.obsm.keys())


def get_spatialdata_label_options_from_sdata(sdata: SpatialData) -> list[SpatialDataLabelsOption]:
    """Return selectable labels elements directly from a loaded SpatialData object."""
    return [
        SpatialDataLabelsOption(
            label_name=label_name,
            display_name=label_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.labels[label_name]),
        )
        for label_name in _get_label_names(sdata)
    ]


def get_coordinate_system_names_from_sdata(sdata: SpatialData) -> list[str]:
    """Return all coordinate-system names exposed by labels and images in a loaded `SpatialData`."""
    coordinate_systems: set[str] = set()

    for label_name in _get_label_names(sdata):
        coordinate_systems.update(_get_element_coordinate_systems(sdata.labels[label_name]))

    for image_name in _get_image_names(sdata):
        coordinate_systems.update(_get_element_coordinate_systems(sdata.images[image_name]))

    return sorted(coordinate_systems)


def get_image_channel_names_from_sdata(sdata: SpatialData, image_name: str) -> list[str]:
    """Return channel names for an image element, or an empty list without a channel axis."""
    if image_name not in _get_image_names(sdata):
        raise ValueError(f"Image element `{image_name}` is not available in the selected SpatialData object.")

    image_element = sdata.images[image_name]
    axes = get_axes_names(image_element)
    if "c" not in axes:
        return []

    if isinstance(image_element, DataArray):
        channel_values = list(image_element.coords.indexes["c"])
    else:
        scale0 = next(iter(image_element["scale0"].values()))
        channel_values = list(scale0.coords.indexes["c"])

    channel_names = [str(channel_value) for channel_value in channel_values]
    duplicates: list[str] = []
    seen: set[str] = set()
    duplicate_seen: set[str] = set()
    for channel_name in channel_names:
        if channel_name in seen and channel_name not in duplicate_seen:
            duplicates.append(channel_name)
            duplicate_seen.add(channel_name)
        seen.add(channel_name)

    if duplicates:
        duplicate_names = ", ".join(f"`{channel_name}`" for channel_name in duplicates)
        raise ValueError(
            f"Image element `{image_name}` exposes duplicate channel names ({duplicate_names}), "
            "which napari-harpy does not support. "
            "Update the channel names in the SpatialData object with "
            "`sdata.set_channel_names(...)`."
        )

    return channel_names


def get_spatialdata_label_options_for_coordinate_system_from_sdata(
    *,
    sdata: SpatialData,
    coordinate_system: str,
) -> list[SpatialDataLabelsOption]:
    """Return labels options restricted to a selected coordinate system."""
    return [
        SpatialDataLabelsOption(
            label_name=label_name,
            display_name=label_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.labels[label_name]),
        )
        for label_name in _get_label_names(sdata)
        if coordinate_system in _get_element_coordinate_systems(sdata.labels[label_name])
    ]


def get_spatialdata_image_options_for_coordinate_system_from_sdata(
    *,
    sdata: SpatialData,
    coordinate_system: str,
) -> list[SpatialDataImageOption]:
    """Return image options restricted to a selected coordinate system."""
    return [
        SpatialDataImageOption(
            image_name=image_name,
            display_name=image_name,
            sdata=sdata,
            coordinate_systems=_get_element_coordinate_systems(sdata.images[image_name]),
        )
        for image_name in _get_image_names(sdata)
        if coordinate_system in _get_element_coordinate_systems(sdata.images[image_name])
    ]


def _get_table_model_attrs(table: AnnData, table_name: str) -> dict[str, Any]:
    attrs = table.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(attrs, dict):
        raise ValueError(f"Table `{table_name}` is missing `{TableModel.ATTRS_KEY}` metadata.")

    required_keys = (TableModel.REGION_KEY_KEY, TableModel.INSTANCE_KEY)
    missing_keys = [key for key in required_keys if key not in attrs]
    if missing_keys:
        missing = ", ".join(f"`{key}`" for key in missing_keys)
        raise ValueError(f"Table `{table_name}` is missing required SpatialData metadata: {missing}.")

    return attrs


def _normalize_table_model_attrs(table: AnnData) -> None:
    attrs = table.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(attrs, dict):
        return

    if TableModel.REGION_KEY in attrs:
        attrs[TableModel.REGION_KEY] = _normalize_region_attr_value(attrs[TableModel.REGION_KEY])

    for key in (TableModel.REGION_KEY_KEY, TableModel.INSTANCE_KEY):
        if key in attrs:
            attrs[key] = _normalize_scalar_string_attr_value(attrs[key])


def _normalize_regions(region: Any) -> tuple[str, ...]:
    if region is None:
        return ()

    if isinstance(region, str):
        return (region,)

    return tuple(_flatten_string_values(region))


def _normalize_region_attr_value(region: Any) -> str | list[str] | None:
    if region is None or isinstance(region, str):
        return region

    return _flatten_string_values(region)


def _normalize_scalar_string_attr_value(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value

    flattened = _flatten_string_values(value)
    if len(flattened) == 1:
        return flattened[0]

    return value


def _flatten_string_values(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, bytes):
        return [value.decode()]

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _flatten_string_values(value.item())

        flattened: list[str] = []
        for item in value.reshape(-1).tolist():
            flattened.extend(_flatten_string_values(item))
        return flattened

    if isinstance(value, np.generic):
        return [str(value.item())]

    if isinstance(value, Sequence):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_string_values(item))
        return flattened

    return [str(value)]


def _layer_indices_align_with_region_view(layer: Any | None, region_view: AnnData, instance_key: str) -> bool | None:
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return None

    layer_indices = _normalize_layer_indices(metadata.get("indices"))
    if layer_indices is None:
        return None

    region_instances = region_view.obs[instance_key]
    if not region_instances.is_unique:
        return False

    return bool(region_instances.isin(layer_indices).all())


def _normalize_layer_metadata_adata(adata: AnnData) -> AnnData:
    from pandas.api.types import CategoricalDtype

    from napari_harpy._annotation import (
        USER_CLASS_COLORS_KEY,
        USER_CLASS_COLUMN,
    )
    from napari_harpy._class_palette import set_class_annotation_state
    from napari_harpy._classifier import PRED_CLASS_COLORS_KEY, PRED_CLASS_COLUMN

    color_keys_to_strip = {USER_CLASS_COLORS_KEY, PRED_CLASS_COLORS_KEY}

    for column_name, colors_key in (
        (USER_CLASS_COLUMN, USER_CLASS_COLORS_KEY),
        (PRED_CLASS_COLUMN, PRED_CLASS_COLORS_KEY),
    ):
        if column_name not in adata.obs:
            continue

        column = adata.obs[column_name]
        needs_category_normalization = not isinstance(column.dtype, CategoricalDtype)
        if needs_category_normalization:
            set_class_annotation_state(
                adata,
                column,
                column_name=column_name,
                colors_key=colors_key,
                keep_colors=False,
            )

    if any(color_key in adata.uns for color_key in color_keys_to_strip):
        adata.uns = {key: value for key, value in adata.uns.items() if key not in color_keys_to_strip}

    return adata


def _normalize_layer_indices(indices: Any) -> list[Any] | None:
    if indices is None or isinstance(indices, str | bytes):
        return None

    if not isinstance(indices, Sequence):
        try:
            indices = list(indices)
        except TypeError:
            return None

    return [index for index in indices if index != 0]


def _is_pickable_labels_layer(layer: Any) -> bool:
    events = getattr(layer, "events", None)
    return isinstance(layer, Labels) and getattr(events, "selected_label", None) is not None


def _get_label_names(sdata: SpatialData) -> list[str]:
    labels = getattr(sdata, "labels", {})
    return sorted(labels.keys())


def _get_image_names(sdata: SpatialData) -> list[str]:
    images = getattr(sdata, "images", {})
    return sorted(images.keys())


def _get_element_coordinate_systems(element: Any) -> tuple[str, ...]:
    return tuple(sorted(get_transformation(element, get_all=True).keys()))


def _get_loaded_spatialdata_layer(
    viewer: Any | None,
    *,
    sdata: SpatialData,
    element_name: str,
    layer_filter: Callable[[Any], bool],
    coordinate_system: str | None = None,
) -> Any | None:
    layers = getattr(viewer, "layers", None)
    if layers is None:
        return None

    for layer in layers:
        metadata = getattr(layer, "metadata", None)
        if not isinstance(metadata, dict):
            continue

        if metadata.get("sdata") is not sdata:
            continue

        if metadata.get("name") != element_name:
            continue

        if coordinate_system is not None:
            layer_coordinate_system = metadata.get("coordinate_system", metadata.get("_current_cs"))
            if layer_coordinate_system is not None and layer_coordinate_system != coordinate_system:
                continue

        if layer_filter(layer):
            return layer

    return None


def build_layer_metadata_adata(
    viewer: Any | None,
    sdata: SpatialData,
    label_name: str,
    table_name: str,
) -> AnnData | None:
    """Build the AnnData stored as napari layer metadata for a labels element.

    Harpy keeps ``sdata[table_name]`` as the authoritative in-memory table.
    The napari layer metadata ``adata`` is only a compatibility cache for
    ``napari-spatialdata``. To avoid materializing a second full AnnData
    object on every refresh, we first try to expose a lightweight view of
    just the rows for ``label_name``.

    That region-only view is safe when its ``instance_key`` values are
    compatible with the loaded layer's cached ``metadata["indices"]``.
    ``napari-spatialdata`` later merges those indices against
    ``adata.obs[instance_key]``, so the cheap path only requires the region
    view to have unique instance ids and for those ids to be present in the
    layer indices. If that compatibility check fails, we fall back to
    ``join_spatialelement_table(..., how="left", match_rows="left")`` so
    the cache matches ``napari-spatialdata``'s original layer-specific
    semantics.

    Harpy no longer uses this helper in the normal widget lifecycle for the
    same reason described in ``refresh_layer_table_metadata(...)``:
    ``napari-spatialdata`` may later rebuild ``layer.metadata["adata"]``
    from the authoritative ``sdata[table_name]`` table again. As a result,
    Harpy now treats ``sdata[table_name]`` as the only source of truth and
    keeps this helper as an explicit cache-building utility rather than a
    path that is relied on during normal interaction.
    """
    table = get_table(sdata, table_name)
    table_metadata = get_table_metadata(sdata, table_name)
    region_mask = (table.obs[table_metadata.region_key] == label_name).to_numpy(dtype=bool, copy=False)
    if not region_mask.any():
        return None

    region_view = table[region_mask, :]
    layer = _get_loaded_spatialdata_layer(
        viewer,
        sdata=sdata,
        element_name=label_name,
        layer_filter=_is_pickable_labels_layer,
    )
    if _layer_indices_align_with_region_view(layer, region_view, table_metadata.instance_key) is not False:
        return _normalize_layer_metadata_adata(region_view)

    _, adata = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=label_name,
        table_name=table_name,
        how="left",
        match_rows="left",
    )
    if adata is None or adata.shape[0] == 0:
        return None

    return _normalize_layer_metadata_adata(adata)


def refresh_layer_table_metadata(viewer: Any | None, sdata: SpatialData, label_name: str, table_name: str) -> bool:
    """Refresh table-derived metadata on the loaded napari layer for a labels element.

    Harpy originally tried to keep ``layer.metadata["adata"]`` refreshed as a
    compatibility cache for ``napari-spatialdata`` so that edits to
    ``sdata[table_name]`` would also be reflected in the currently loaded
    labels layer metadata.

    In practice, ``napari-spatialdata`` may later rebuild and overwrite that
    cache from the authoritative ``sdata[table_name]`` table again when its
    own widgets update. Because of that, Harpy no longer calls this helper as
    part of the normal widget lifecycle and instead treats
    ``sdata[table_name]`` as the only source of truth. This method is kept as
    an explicit best-effort utility for cases where refreshing the layer
    metadata cache is still useful.
    """
    layer = _get_loaded_spatialdata_layer(
        viewer,
        sdata=sdata,
        element_name=label_name,
        layer_filter=_is_pickable_labels_layer,
    )
    if layer is None:
        return False

    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        layer.metadata = metadata

    table_metadata = get_table_metadata(sdata, table_name)
    layer.metadata["adata"] = build_layer_metadata_adata(viewer, sdata, label_name, table_name)
    layer.metadata["region_key"] = table_metadata.region_key
    layer.metadata["instance_key"] = table_metadata.instance_key

    table_names = get_annotating_table_names(sdata, label_name)
    layer.metadata["table_names"] = table_names if table_names else None
    return True
