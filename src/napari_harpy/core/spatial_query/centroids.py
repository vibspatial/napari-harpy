from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd
from harpy.image import get_dataarray
from harpy.utils import RasterAggregator

from napari_harpy.core.spatial_query.canonical import (
    CANONICAL_OBSM_KEY,
    apply_canonical_cache_update,
    build_canonical_cache_update_payload,
    build_canonical_source_signature,
    inspect_canonical_cache,
)
from napari_harpy.core.spatial_query.canonical_models import (
    CanonicalCacheReport,
    CanonicalCacheState,
    CanonicalCacheUpdatePayload,
    CanonicalCentersResult,
    CanonicalRegionBinding,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData


def calculate_canonical_centers(
    sdata: SpatialData,
    report: CanonicalCacheReport,
) -> CanonicalCacheUpdatePayload:
    """Calculate selected-region centers without mutating the canonical cache."""
    if not isinstance(report, CanonicalCacheReport):
        raise TypeError("Canonical center calculation requires a CanonicalCacheReport.")

    labels_name = report.labels_name
    current_source = build_canonical_source_signature(sdata, labels_name)
    if current_source != report.source_signature:
        raise ValueError("Labels source changed after canonical cache inspection; calculation was rejected.")

    labels = get_dataarray(sdata, labels_name, scale="scale0").data
    _validate_instance_ids_fit_labels_dtype(report.binding, labels.dtype)
    centers = _calculate_centers_with_raster_aggregator(labels, report.binding)
    return build_canonical_cache_update_payload(
        binding=report.binding,
        centers=centers,
        source_signature=report.source_signature,
    )


def ensure_canonical_centers(
    sdata: SpatialData,
    *,
    table_name: str,
    labels_name: str,
    force_recalculation: bool = False,
) -> CanonicalCentersResult:
    """Return selected-region centers, reusing or updating the cache as needed."""
    if not isinstance(force_recalculation, bool):
        raise TypeError("force_recalculation must be a boolean.")

    report = inspect_canonical_cache(
        sdata,
        table_name=table_name,
        labels_name=labels_name,
    )
    if report.state is CanonicalCacheState.VALID and not force_recalculation:
        table = sdata.tables[table_name]
        centers = np.asarray(table.obsm[CANONICAL_OBSM_KEY])[report.binding.row_positions]
        return CanonicalCentersResult(
            source_signature=report.source_signature,
            binding=report.binding,
            centers=centers,
            cache_update=None,
        )

    payload = calculate_canonical_centers(sdata, report)
    cache_update = apply_canonical_cache_update(sdata, payload)
    return CanonicalCentersResult(
        source_signature=payload.source_signature,
        binding=payload.binding,
        centers=payload.centers,
        cache_update=cache_update,
    )


def _validate_instance_ids_fit_labels_dtype(binding: CanonicalRegionBinding, labels_dtype: np.dtype) -> None:
    dtype = np.dtype(labels_dtype)
    maximum = int(np.iinfo(dtype).max)
    instance_ids = binding.instance_ids
    too_large = instance_ids > maximum
    if np.any(too_large):
        invalid_ids = instance_ids[too_large]
        raise ValueError(
            f"{len(invalid_ids)} selected instance ID(s) cannot be represented by labels dtype `{dtype.name}`"
            f" ({_format_id_preview(invalid_ids)})."
        )


def _calculate_centers_with_raster_aggregator(
    labels: da.Array,
    binding: CanonicalRegionBinding,
) -> np.ndarray:
    aggregator = RasterAggregator(
        mask_dask_array=labels[None, ...],
        image_dask_array=None,
        instance_key=binding.instance_key,
        run_on_gpu=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        result = aggregator.center_of_mass(index=binding.instance_ids)
    if not isinstance(result, pd.DataFrame):
        raise ValueError("RasterAggregator center_of_mass must return a pandas DataFrame.")
    coordinate_columns = [0, 1, 2]
    expected_columns = [*coordinate_columns, binding.instance_key]
    if result.columns.tolist() != expected_columns:
        raise ValueError(
            "RasterAggregator center_of_mass must return z, y, x, and instance ID columns in that order."
        )

    output_ids = result[binding.instance_key].to_numpy()
    if output_ids.dtype != binding.instance_ids.dtype or not np.array_equal(output_ids, binding.instance_ids):
        raise ValueError("RasterAggregator center_of_mass instance IDs must match the requested IDs in order.")

    try:
        centers = result[coordinate_columns].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("RasterAggregator center_of_mass coordinates must be numeric.") from exc
    finite = np.isfinite(centers).all(axis=1)
    if not finite.all():
        missing_ids = binding.instance_ids[~finite]
        raise ValueError(
            f"Labels element `{binding.labels_name}` has no finite center for {len(missing_ids)} requested instance"
            f" ID(s) ({_format_id_preview(missing_ids)})."
        )
    return centers


def _format_id_preview(instance_ids: np.ndarray, limit: int = 5) -> str:
    preview = ", ".join(str(int(value)) for value in instance_ids[:limit])
    if len(instance_ids) > limit:
        preview += ", ..."
    return preview
