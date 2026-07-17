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
    """Canonical-center provenance and sorted IDs intersecting the annotation.

    Parameters
    ----------
    canonical_centers
        Complete immutable selected-region centers and provenance used by the
        containment query.
    matched_instance_ids
        Unique, ascending instance IDs whose canonical centers intersect the
        annotation polygons. This is a subset of
        ``canonical_centers.binding.instance_ids``, which contains every
        eligible instance evaluated by the query.

    Notes
    -----
    ``canonical_centers`` intentionally retains the complete immutable center
    result rather than only its binding. The matching instance IDs are a
    geometric decision made from that exact source signature and center
    snapshot. The ``apply_spatial_annotation()`` domain operation in
    ``napari_harpy.core.spatial_query.annotation`` later passes this result to
    ``_require_current_query_provenance()``, which compares both against the
    current cache. This prevents an old query from annotating rows after centers
    were rebuilt or otherwise changed while the result was awaiting review.
    Keeping the existing result object avoids duplicating provenance fields or
    copying its center array.
    """

    canonical_centers: CanonicalCentersResult
    matched_instance_ids: NDArray[np.integer] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_centers, CanonicalCentersResult):
            raise TypeError("Canonical-center query result requires a CanonicalCentersResult.")
        matched_instance_ids = _readonly_integer_ids(self.matched_instance_ids)
        if len(matched_instance_ids) > 1 and np.any(matched_instance_ids[1:] <= matched_instance_ids[:-1]):
            raise ValueError("Canonical-center query result instance IDs must be unique and sorted.")
        if (
            len(matched_instance_ids)
            and not np.isin(matched_instance_ids, self.binding.instance_ids, assume_unique=True).all()
        ):
            raise ValueError("Canonical-center query result instance IDs must be a subset of the selected region.")
        object.__setattr__(self, "matched_instance_ids", matched_instance_ids)

    @property
    def binding(self) -> CanonicalRegionBinding:
        """Return the selected-region binding used by the query."""
        return self.canonical_centers.binding

    @property
    def eligible_instance_count(self) -> int:
        """Return the number of instances evaluated by the query."""
        return self.binding.n_obs

    @property
    def matched_instance_count(self) -> int:
        """Return the number of matching instances."""
        return len(self.matched_instance_ids)
