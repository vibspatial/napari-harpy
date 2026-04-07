from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spatialdata import get_element_annotators
from spatialdata.models import TableModel

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

    @property
    def identity(self) -> tuple[int, str]:
        """Return a stable identity for preserving widget selection across refreshes."""
        return (id(self.sdata), self.label_name)


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


class SpatialDataAdapter:
    """Thin wrapper around viewer-linked SpatialData discovery.

    The adapter intentionally stays lightweight. It centralizes the small amount
    of `napari-spatialdata` / `spatialdata` integration we need today while
    still delegating table semantics to `spatialdata` itself.
    """

    def __init__(self, viewer: Any | None = None) -> None:
        self._viewer = viewer

    def get_label_options(self) -> list[SpatialDataLabelsOption]:
        """Collect selectable labels elements from the current viewer."""
        return _get_spatialdata_label_options_from_viewer(self._viewer)

    def get_annotating_table_names(self, sdata: SpatialData, label_name: str) -> list[str]:
        """Return the tables that annotate a labels element."""
        return sorted(get_element_annotators(sdata, label_name))

    def get_table(self, sdata: SpatialData, table_name: str) -> AnnData:
        """Return a validated annotating table."""
        table = sdata[table_name]
        return TableModel.validate(table)

    def get_table_metadata(self, sdata: SpatialData, table_name: str) -> SpatialDataTableMetadata:
        """Resolve table linkage metadata from `TableModel` attributes."""
        table = self.get_table(sdata, table_name)
        attrs = _get_table_model_attrs(table, table_name)

        return SpatialDataTableMetadata(
            table_name=table_name,
            region_key=str(attrs[TableModel.REGION_KEY_KEY]),
            instance_key=str(attrs[TableModel.INSTANCE_KEY]),
            regions=_normalize_regions(attrs.get(TableModel.REGION_KEY)),
        )

    def get_table_obsm_keys(self, sdata: SpatialData, table_name: str) -> list[str]:
        """Return available feature matrix keys from `adata.obsm`."""
        table = self.get_table(sdata, table_name)
        return sorted(table.obsm.keys())


def get_spatialdata_label_options(viewer: Any | None) -> list[SpatialDataLabelsOption]:
    """Collect selectable labels elements from viewer layers linked by `napari-spatialdata`.

    This helper scans the current napari viewer for layers whose metadata contains an
    associated `SpatialData` object under `layer.metadata["sdata"]`, as provided by
    `napari-spatialdata`. The discovered datasets are deduplicated, and all available
    entries from `sdata.labels` are returned as `SpatialDataLabelsOption` objects.

    The returned options carry both the label name and the originating `SpatialData`
    object so the widget can safely support viewers containing multiple datasets.
    """
    return SpatialDataAdapter(viewer).get_label_options()


def get_annotating_table_names(sdata: SpatialData, label_name: str) -> list[str]:
    """Return the table names that annotate a labels element in a SpatialData object."""
    return SpatialDataAdapter().get_annotating_table_names(sdata, label_name)


def get_table_metadata(sdata: SpatialData, table_name: str) -> SpatialDataTableMetadata:
    """Return linkage metadata for an annotating table."""
    return SpatialDataAdapter().get_table_metadata(sdata, table_name)


def get_table_obsm_keys(sdata: SpatialData, table_name: str) -> list[str]:
    """Return the available feature matrix keys from `adata.obsm` for a table in a SpatialData object."""
    return SpatialDataAdapter().get_table_obsm_keys(sdata, table_name)


def _get_spatialdata_label_options_from_viewer(viewer: Any | None) -> list[SpatialDataLabelsOption]:
    if viewer is None:
        return []

    layers = getattr(viewer, "layers", None)
    if layers is None:
        return []

    sdatas = _get_unique_spatialdata_objects(layers)
    if not sdatas:
        return []

    label_name_counts = Counter(label_name for sdata in sdatas for label_name in _get_label_names(sdata))

    options: list[SpatialDataLabelsOption] = []
    multiple_sdatas = len(sdatas) > 1

    for index, sdata in enumerate(sdatas, start=1):
        dataset_name = _get_dataset_name(sdata, index)
        for label_name in _get_label_names(sdata):
            # Build the text shown in the combo box. We keep the raw
            # `label_name` for program logic and only append the dataset name
            # when the user may need help distinguishing otherwise identical labels.
            display_name = label_name
            if multiple_sdatas or label_name_counts[label_name] > 1:
                display_name = f"{label_name} ({dataset_name})"

            options.append(
                SpatialDataLabelsOption(
                    label_name=label_name,
                    display_name=display_name,
                    sdata=sdata,
                )
            )

    return options


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


def _normalize_regions(region: str | list[str] | None) -> tuple[str, ...]:
    if region is None:
        return ()

    if isinstance(region, str):
        return (region,)

    return tuple(str(label_name) for label_name in region)


def _get_unique_spatialdata_objects(layers: Any) -> list[SpatialData]:
    unique_sdatas: list[SpatialData] = []
    seen_ids: set[int] = set()

    for layer in layers:
        metadata = getattr(layer, "metadata", None)
        if not isinstance(metadata, dict):
            continue

        sdata = metadata.get("sdata")
        if sdata is None or not hasattr(sdata, "labels"):
            continue

        sdata_id = id(sdata)
        if sdata_id in seen_ids:
            continue

        seen_ids.add(sdata_id)
        unique_sdatas.append(sdata)

    return unique_sdatas


def _get_label_names(sdata: SpatialData) -> list[str]:
    labels = getattr(sdata, "labels", {})
    return sorted(labels.keys())


def _get_dataset_name(sdata: SpatialData, index: int) -> str:
    for attr in ("path", "_path", "name"):
        value = getattr(sdata, attr, None)
        if value:
            return Path(str(value)).name

    return f"SpatialData {index}"
