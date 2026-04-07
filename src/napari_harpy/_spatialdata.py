from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatialdata import SpatialData


@dataclass(frozen=True)
class SpatialDataLabelsOption:
    """A selectable labels element discovered from a viewer-linked SpatialData object."""

    label_name: str
    display_name: str
    dataset_name: str
    sdata: SpatialData

    @property
    def identity(self) -> tuple[int, str]:
        """Return a stable identity for preserving widget selection across refreshes."""
        return (id(self.sdata), self.label_name)


def get_spatialdata_label_options(viewer: Any | None) -> list[SpatialDataLabelsOption]:
    """Collect labels elements exposed by `napari-spatialdata` in the current viewer."""
    if viewer is None:
        return []

    layers = getattr(viewer, "layers", None)
    if layers is None:
        return []

    sdatas = _get_unique_spatialdata_objects(layers)
    if not sdatas:
        return []

    label_name_counts = Counter(
        label_name
        for sdata in sdatas
        for label_name in _get_label_names(sdata)
    )

    options: list[SpatialDataLabelsOption] = []
    multiple_sdatas = len(sdatas) > 1

    for index, sdata in enumerate(sdatas, start=1):
        dataset_name = _get_dataset_name(sdata, index)
        for label_name in _get_label_names(sdata):
            display_name = label_name
            if multiple_sdatas or label_name_counts[label_name] > 1:
                display_name = f"{label_name} ({dataset_name})"

            options.append(
                SpatialDataLabelsOption(
                    label_name=label_name,
                    display_name=display_name,
                    dataset_name=dataset_name,
                    sdata=sdata,
                )
            )

    return options


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
