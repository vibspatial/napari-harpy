from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from napari_harpy.core.spatial_query.canonical_models import (
    CanonicalCentersResult,
    CanonicalRegionBinding,
    _readonly_array,
    _readonly_integer_ids,
)


@dataclass(frozen=True)
class CanonicalCenterQueryRequest:
    """Immutable computational inputs for one canonical-center query.

    Parameters
    ----------
    canonical_centers
        Selected-region instance identity and canonical centers in the labels
        element's intrinsic coordinate frame.
    polygons
        Validated Polygon snapshots in the source Shapes element's intrinsic
        x/y coordinate frame.
    polygons_to_labels_affine
        Homogeneous 3 x 3 affine mapping polygon x/y coordinates into the
        selected labels element's intrinsic x/y coordinate frame.
    """

    canonical_centers: CanonicalCentersResult
    polygons: tuple[Polygon, ...] = field(repr=False, compare=False)
    polygons_to_labels_affine: NDArray[np.float64] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_centers, CanonicalCentersResult):
            raise TypeError("Canonical-center query requires a CanonicalCentersResult.")
        if self.canonical_centers.source_signature.dims != ("y", "x"):
            raise ValueError("Canonical-center containment queries require a two-dimensional labels source.")

        polygons = tuple(self.polygons)
        if not polygons:
            raise ValueError("Canonical-center query requires at least one Polygon geometry.")
        if any(not isinstance(polygon, Polygon) for polygon in polygons):
            raise TypeError("Canonical-center query geometries must be Shapely Polygon objects.")

        affine = _readonly_array(self.polygons_to_labels_affine, dtype=np.float64)
        if affine.shape != (3, 3):
            raise ValueError("Polygons-to-labels affine must be a 3 x 3 homogeneous matrix.")
        if not np.isfinite(affine).all():
            raise ValueError("Polygons-to-labels affine must contain only finite values.")

        object.__setattr__(self, "polygons", polygons)
        object.__setattr__(self, "polygons_to_labels_affine", affine)

    @property
    def table_name(self) -> str:
        """Return the table name from the canonical-center result."""
        return self.canonical_centers.table_name

    @property
    def labels_name(self) -> str:
        """Return the labels name from the canonical-center result."""
        return self.canonical_centers.labels_name


@dataclass(frozen=True)
class CanonicalCenterQueryResult:
    """Sorted instance IDs whose canonical centers intersect the annotation."""

    binding: CanonicalRegionBinding
    instance_ids: NDArray[np.integer] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, CanonicalRegionBinding):
            raise TypeError("Canonical-center query result requires a CanonicalRegionBinding.")
        instance_ids = _readonly_integer_ids(self.instance_ids)
        if len(instance_ids) > 1 and np.any(instance_ids[1:] <= instance_ids[:-1]):
            raise ValueError("Canonical-center query result instance IDs must be unique and sorted.")
        object.__setattr__(self, "instance_ids", instance_ids)

    @property
    def eligible_instance_count(self) -> int:
        """Return the number of instances evaluated by the query."""
        return self.binding.n_obs

    @property
    def matched_instance_count(self) -> int:
        """Return the number of matching instances."""
        return len(self.instance_ids)
