from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, NoReturn

import numpy as np
from scipy.sparse import issparse
from xarray import DataArray

from napari_harpy.core.spatial_query.models import (
    CanonicalCacheMismatch,
    CanonicalCacheReport,
    CanonicalCacheState,
    CanonicalInstallAction,
    CanonicalInstallationPayload,
    CanonicalInstallationResult,
    CanonicalMetadata,
    CanonicalMismatchCode,
    CanonicalRegionBinding,
    CanonicalRegionMetadata,
    CanonicalSourceSignature,
)
from napari_harpy.core.spatialdata import SpatialDataTableMetadata, get_table_metadata

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

CANONICAL_OBSM_KEY = "spatial_canonical"
SPATIAL_COORDINATES_KEY = "spatial_coordinates"
CANONICAL_SCHEMA_VERSION = 1
CANONICAL_ALGORITHM_VERSION = 1
CANONICAL_CHANGED_PATHS = (
    "obsm/spatial_canonical",
    "uns/spatial_coordinates/spatial_canonical",
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "obsm_key",
    "axes",
    "dtype",
    "region_key",
    "instance_key",
    "regions",
}
_REGION_KEYS = {
    "source_element",
    "source_element_type",
    "source_scale",
    "coordinate_frame",
    "calculation",
    "coverage",
    "source",
    "generated_by",
}


class _UnsupportedSchemaError(ValueError):
    pass


class _TopLevelContractError(ValueError):
    pass


class _RegionMetadataError(ValueError):
    pass


def build_canonical_source_signature(sdata: SpatialData, labels_name: str) -> CanonicalSourceSignature:
    """Build the structural scale0 signature without reading labels pixels."""
    if not isinstance(labels_name, str) or not labels_name:
        raise ValueError("Labels name must be a non-empty string.")
    labels = getattr(sdata, "labels", None)
    if labels is None or labels_name not in labels:
        raise ValueError(f"Labels element `{labels_name}` is not available in the selected SpatialData object.")

    element = labels[labels_name]
    if isinstance(element, DataArray):
        scale0 = element
    else:
        try:
            scale0_node = element["scale0"]
            scale0 = next(iter(scale0_node.values()))
        except (KeyError, StopIteration, TypeError, AttributeError) as exc:
            raise ValueError(f"Labels element `{labels_name}` does not expose a readable `scale0` DataArray.") from exc
        if not isinstance(scale0, DataArray):
            raise ValueError(f"Labels element `{labels_name}` scale0 is not an xarray DataArray.")

    dims = tuple(str(dim) for dim in scale0.dims)
    shape = tuple(int(size) for size in scale0.shape)
    try:
        dtype = np.dtype(scale0.dtype)
    except TypeError as exc:
        raise ValueError(f"Labels element `{labels_name}` exposes an unsupported dtype.") from exc
    if dtype.kind not in "iu":
        raise ValueError(f"Labels element `{labels_name}` must use an integer dtype.")
    if dims != ("y", "x"):
        raise ValueError("Canonical metadata schema version 1 requires source dims (`y`, `x`).")

    return CanonicalSourceSignature(
        labels_name=labels_name,
        source_scale="scale0",
        dims=dims,
        shape=shape,
        dtype=dtype.name,
    )


def build_canonical_region_binding(
    table: AnnData,
    table_metadata: SpatialDataTableMetadata,
    labels_name: str,
) -> CanonicalRegionBinding:
    """Validate and capture one selected-region table binding."""
    if not table_metadata.annotates(labels_name):
        raise ValueError(f"Table `{table_metadata.table_name}` does not declare labels region `{labels_name}`.")
    for key in (table_metadata.region_key, table_metadata.instance_key):
        if key not in table.obs.columns:
            raise ValueError(f"Table `{table_metadata.table_name}` is missing required obs column `{key}`.")

    region_values = table.obs[table_metadata.region_key]
    mask = np.asarray(region_values == labels_name, dtype=bool)
    row_positions = np.flatnonzero(mask).astype(np.intp, copy=False)
    if len(row_positions) == 0:
        raise ValueError(f"Table `{table_metadata.table_name}` contains no rows for labels region `{labels_name}`.")

    try:
        return CanonicalRegionBinding(
            labels_name=labels_name,
            region_key=table_metadata.region_key,
            instance_key=table_metadata.instance_key,
            row_positions=row_positions,
            instance_ids=table.obs.iloc[row_positions][table_metadata.instance_key].to_numpy(),
        )
    except ValueError as exc:
        if "unique" not in str(exc):
            raise
        raise ValueError(
            f"Table `{table_metadata.table_name}` contains duplicate `{table_metadata.instance_key}` values "
            f"within labels region `{labels_name}`."
        ) from exc


def build_canonical_metadata(
    *,
    region_key: str,
    instance_key: str,
    regions: Mapping[str, CanonicalRegionMetadata],
    schema_version: int = CANONICAL_SCHEMA_VERSION,
) -> CanonicalMetadata:
    """Build typed canonical metadata for storage."""
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f"Only canonical schema version {CANONICAL_SCHEMA_VERSION} can be built.")
    for region_metadata in regions.values():
        _validate_schema_v1_source(region_metadata.source_signature)
    return CanonicalMetadata(
        schema_version=schema_version,
        region_key=region_key,
        instance_key=instance_key,
        regions=regions,
    )


def canonical_metadata_to_storage(metadata: CanonicalMetadata) -> dict[str, object]:
    """Serialize typed metadata using AnnData-zarr-compatible values."""
    if metadata.schema_version != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported canonical schema version {metadata.schema_version}.")
    regions: dict[str, object] = {}
    for region, region_metadata in metadata.regions.items():
        source = region_metadata.source_signature
        entry: dict[str, object] = {
            "source_element": region,
            "source_element_type": "labels",
            "source_scale": "scale0",
            "coordinate_frame": {
                "type": "element_intrinsic",
                "element": region,
                "axes": ["x", "y"],
            },
            "calculation": {
                "method": "center_of_mass",
                "weighting": "uniform_label_pixels",
                "background_value": 0,
                "pixel_coordinate_convention": "integer_indices_are_pixel_centers",
                "implementation": "harpy.utils.RasterAggregator.center_of_mass",
                "algorithm_version": region_metadata.algorithm_version,
            },
            "coverage": {
                "scope": "all_rows_for_region",
                "n_obs": region_metadata.n_obs,
                "instance_set_digest": region_metadata.instance_set_digest,
            },
            "source": {
                "element_path": f"labels/{region}",
                "dims": list(source.dims),
                "shape": list(source.shape),
                "dtype": source.dtype,
            },
        }
        generated_by = _serialize_generated_by(region_metadata)
        if generated_by is not None:
            entry["generated_by"] = generated_by
        regions[region] = entry

    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "obsm_key": CANONICAL_OBSM_KEY,
        "axes": ["x", "y"],
        "dtype": "float64",
        "region_key": metadata.region_key,
        "instance_key": metadata.instance_key,
        "regions": regions,
    }


def parse_canonical_metadata(value: object) -> CanonicalMetadata:
    """Strictly parse the canonical schema-v1 storage representation."""
    mapping = _require_mapping(value, "canonical metadata")
    if "schema_version" not in mapping:
        raise _TopLevelContractError("canonical metadata is missing `schema_version`.")
    schema_version = _require_integer(mapping["schema_version"], "schema_version")
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise _UnsupportedSchemaError(f"Unsupported canonical schema version {schema_version}.")
    _require_exact_keys(mapping, _TOP_LEVEL_KEYS, "canonical metadata", _TopLevelContractError)

    _require_equal(mapping["obsm_key"], CANONICAL_OBSM_KEY, "obsm_key")
    _require_string_sequence(mapping["axes"], ("x", "y"), "axes", _TopLevelContractError)
    _require_equal(mapping["dtype"], "float64", "dtype")
    region_key = _require_nonempty_string(mapping["region_key"], "region_key")
    instance_key = _require_nonempty_string(mapping["instance_key"], "instance_key")
    raw_regions = _require_mapping(mapping["regions"], "regions", _RegionMetadataError)

    regions: dict[str, CanonicalRegionMetadata] = {}
    for raw_region, raw_entry in raw_regions.items():
        if not isinstance(raw_region, str) or not raw_region:
            raise _RegionMetadataError("Region names must be non-empty strings.")
        regions[raw_region] = _parse_region_metadata(raw_region, raw_entry)
    return CanonicalMetadata(
        schema_version=schema_version,
        region_key=region_key,
        instance_key=instance_key,
        regions=regions,
    )


def inspect_canonical_cache(
    sdata: SpatialData,
    *,
    table_name: str,
    labels_name: str,
) -> CanonicalCacheReport:
    """Inspect the selected region's canonical cache without mutating stored state."""
    table = sdata[table_name]
    table_metadata = get_table_metadata(sdata, table_name)
    source_signature = build_canonical_source_signature(sdata, labels_name)
    binding = build_canonical_region_binding(table, table_metadata, labels_name)

    matrix_exists = CANONICAL_OBSM_KEY in table.obsm
    registry = table.uns.get(SPATIAL_COORDINATES_KEY)
    metadata_exists = isinstance(registry, Mapping) and CANONICAL_OBSM_KEY in registry

    if not matrix_exists and not metadata_exists:
        return _report(CanonicalCacheState.ABSENT, labels_name, None, source_signature, binding)
    if matrix_exists and not metadata_exists:
        return _report(
            CanonicalCacheState.INVALID,
            labels_name,
            None,
            source_signature,
            binding,
            _pair_mismatch(CanonicalMismatchCode.MATRIX_WITHOUT_METADATA),
        )
    if metadata_exists and not matrix_exists:
        return _report(
            CanonicalCacheState.INVALID,
            labels_name,
            None,
            source_signature,
            binding,
            _pair_mismatch(CanonicalMismatchCode.METADATA_WITHOUT_MATRIX),
        )

    matrix = _validate_canonical_matrix(table.obsm[CANONICAL_OBSM_KEY], table.n_obs)
    if isinstance(matrix, str):
        return _report(
            CanonicalCacheState.INVALID,
            labels_name,
            None,
            source_signature,
            binding,
            _pair_mismatch(CanonicalMismatchCode.MATRIX_INVALID, matrix),
        )

    raw_metadata = registry[CANONICAL_OBSM_KEY]  # type: ignore[index]
    try:
        metadata = parse_canonical_metadata(raw_metadata)
    except _UnsupportedSchemaError as exc:
        return _report(
            CanonicalCacheState.INVALID,
            labels_name,
            None,
            source_signature,
            binding,
            _pair_mismatch(CanonicalMismatchCode.SCHEMA_VERSION_UNSUPPORTED, str(exc)),
        )
    except _TopLevelContractError as exc:
        return _report(
            CanonicalCacheState.INVALID,
            labels_name,
            None,
            source_signature,
            binding,
            _pair_mismatch(CanonicalMismatchCode.TOP_LEVEL_CONTRACT_MISMATCH, str(exc)),
        )
    except _RegionMetadataError as exc:
        return _report(
            CanonicalCacheState.INVALID,
            labels_name,
            None,
            source_signature,
            binding,
            _pair_mismatch(CanonicalMismatchCode.REGION_METADATA_INVALID, str(exc)),
        )
    except (TypeError, ValueError, KeyError) as exc:
        return _report(
            CanonicalCacheState.INVALID,
            labels_name,
            None,
            source_signature,
            binding,
            _pair_mismatch(CanonicalMismatchCode.METADATA_INVALID, str(exc)),
        )

    if metadata.region_key != table_metadata.region_key or metadata.instance_key != table_metadata.instance_key:
        return _report(
            CanonicalCacheState.INVALID,
            labels_name,
            metadata,
            source_signature,
            binding,
            _pair_mismatch(
                CanonicalMismatchCode.TOP_LEVEL_CONTRACT_MISMATCH,
                "Canonical linkage keys do not match the current SpatialData table metadata.",
            ),
        )

    region_metadata = metadata.regions.get(labels_name)
    if region_metadata is None:
        return _report(
            CanonicalCacheState.PARTIAL,
            labels_name,
            metadata,
            source_signature,
            binding,
            _region_mismatch(CanonicalMismatchCode.REGION_NOT_REGISTERED, labels_name),
        )

    mismatches: list[CanonicalCacheMismatch] = []
    if region_metadata.source_signature != source_signature:
        mismatches.append(_region_mismatch(CanonicalMismatchCode.SOURCE_SIGNATURE_MISMATCH, labels_name))
    if (
        region_metadata.n_obs != binding.n_obs
        or region_metadata.instance_set_digest != binding.instance_set_digest
    ):
        mismatches.append(_region_mismatch(CanonicalMismatchCode.TABLE_SIGNATURE_MISMATCH, labels_name))
    if region_metadata.algorithm_version != CANONICAL_ALGORITHM_VERSION:
        mismatches.append(_region_mismatch(CanonicalMismatchCode.ALGORITHM_VERSION_MISMATCH, labels_name))
    if not np.isfinite(matrix[binding.row_positions]).all():
        mismatches.append(_region_mismatch(CanonicalMismatchCode.REGION_COORDINATES_INVALID, labels_name))

    return _report(
        CanonicalCacheState.STALE if mismatches else CanonicalCacheState.VALID,
        labels_name,
        metadata,
        source_signature,
        binding,
        *mismatches,
    )


def build_canonical_installation_payload(
    *,
    table_name: str,
    binding: CanonicalRegionBinding,
    centers_xy: object,
    source_signature: CanonicalSourceSignature,
) -> CanonicalInstallationPayload:
    """Validate calculated centers and capture them in an immutable payload."""
    return CanonicalInstallationPayload(
        table_name=table_name,
        binding=binding,
        centers_xy=centers_xy,
        source_signature=source_signature,
    )


def install_canonical_cache(sdata: SpatialData, payload: CanonicalInstallationPayload) -> CanonicalInstallationResult:
    """Atomically install already-calculated coordinates after fresh validation.

    This function never calculates centroids. The intended orchestration flow is::

        ensure canonical cache
            ↓
        inspect_canonical_cache()
            ↓
        VALID → reuse existing coordinates
            ↓ otherwise
        calculate centroids
            ↓
        build_canonical_installation_payload() # validate/capture result
            ↓
        install_canonical_cache()

    Installation inspects the cache again because the labels source or table
    binding may have changed while centroids were being calculated. This
    fresh inspection rejects an outdated payload and determines the safe
    create, extend, refresh, or rebuild action before mutating the cache.
    """
    labels_name = payload.binding.labels_name
    report = inspect_canonical_cache(
        sdata,
        table_name=payload.table_name,
        labels_name=labels_name,
    )
    if payload.source_signature != report.source_signature:
        raise ValueError("Labels source changed after canonical centers were calculated; installation was rejected.")
    if payload.binding != report.binding:
        raise ValueError(
            "Table region binding changed after canonical centers were calculated; installation was rejected."
        )

    table = sdata[payload.table_name]
    table_metadata = get_table_metadata(sdata, payload.table_name)
    existing_matrix = table.obsm.get(CANONICAL_OBSM_KEY)
    candidate_matrix = np.full((table.n_obs, 2), np.nan, dtype=np.float64)
    preserved_regions: dict[str, CanonicalRegionMetadata] = {}

    # Cache installation flow by CanonicalCacheState:
    #
    # ABSENT
    #     → start with NaNs
    #     → install selected region
    #
    # INVALID
    #     → start with NaNs
    #     → trust no existing region
    #     → install selected region
    #
    # PARTIAL / STALE / VALID
    #     → start with NaNs
    #     → copy every independently revalidated other region
    #     → install or replace selected region
    if report.state in (CanonicalCacheState.PARTIAL, CanonicalCacheState.STALE, CanonicalCacheState.VALID):
        assert report.metadata is not None
        assert existing_matrix is not None
        preserved_regions = _preserve_valid_other_regions(
            sdata,
            table,
            table_metadata,
            report.metadata,
            np.asarray(existing_matrix),
            selected_region=labels_name,
            candidate_matrix=candidate_matrix,
        )

    sorted_payload_positions = np.argsort(payload.binding.instance_ids)
    sorted_payload_ids = payload.binding.instance_ids[sorted_payload_positions]
    selected_positions = np.searchsorted(sorted_payload_ids, report.binding.instance_ids)
    if np.any(selected_positions >= len(sorted_payload_ids)) or not np.array_equal(
        sorted_payload_ids[selected_positions], report.binding.instance_ids
    ):
        raise ValueError("Canonical payload does not contain every current selected-region instance ID.")
    candidate_matrix[report.binding.row_positions] = payload.centers_xy[sorted_payload_positions[selected_positions]]

    preserved_regions[labels_name] = CanonicalRegionMetadata(
        source_signature=report.source_signature,
        n_obs=report.binding.n_obs,
        instance_set_digest=report.binding.instance_set_digest,
        algorithm_version=CANONICAL_ALGORITHM_VERSION,
    )
    candidate_metadata = build_canonical_metadata(
        region_key=table_metadata.region_key,
        instance_key=table_metadata.instance_key,
        regions=preserved_regions,
    )
    candidate_storage = canonical_metadata_to_storage(candidate_metadata)
    matrix_error = _validate_canonical_matrix(candidate_matrix, table.n_obs)
    if isinstance(matrix_error, str):  # pragma: no cover - defensive invariant
        raise RuntimeError(f"Constructed an invalid canonical matrix: {matrix_error}")
    parse_canonical_metadata(candidate_storage)

    action = {
        CanonicalCacheState.ABSENT: CanonicalInstallAction.CREATE,
        CanonicalCacheState.PARTIAL: CanonicalInstallAction.EXTEND,
        CanonicalCacheState.STALE: CanonicalInstallAction.REFRESH,
        CanonicalCacheState.INVALID: CanonicalInstallAction.REBUILD,
        CanonicalCacheState.VALID: CanonicalInstallAction.REFRESH,
    }[report.state]
    _assign_canonical_pair_atomically(table, candidate_matrix, candidate_storage)
    return CanonicalInstallationResult(
        table_name=payload.table_name,
        labels_name=labels_name,
        action=action,
        previous_state=report.state,
        n_installed_rows=len(payload.binding.instance_ids),
        mismatches=report.mismatches,
        changed_paths=CANONICAL_CHANGED_PATHS,
    )


def _parse_region_metadata(
    region: str,
    value: object,
) -> CanonicalRegionMetadata:
    entry = _require_mapping(value, f"region `{region}`", _RegionMetadataError)
    required_keys = _REGION_KEYS - {"generated_by"}
    actual_keys = set(entry)
    if not required_keys.issubset(actual_keys) or not actual_keys.issubset(_REGION_KEYS):
        _raise_key_error(entry, required_keys, _REGION_KEYS, f"region `{region}`", _RegionMetadataError)
    _require_equal(entry["source_element"], region, "source_element", _RegionMetadataError)
    _require_equal(entry["source_element_type"], "labels", "source_element_type", _RegionMetadataError)
    _require_equal(entry["source_scale"], "scale0", "source_scale", _RegionMetadataError)

    coordinate_frame = _require_mapping(entry["coordinate_frame"], "coordinate_frame", _RegionMetadataError)
    _require_exact_keys(
        coordinate_frame,
        {"type", "element", "axes"},
        "coordinate_frame",
        _RegionMetadataError,
    )
    _require_equal(coordinate_frame["type"], "element_intrinsic", "coordinate_frame.type", _RegionMetadataError)
    _require_equal(coordinate_frame["element"], region, "coordinate_frame.element", _RegionMetadataError)
    _require_string_sequence(
        coordinate_frame["axes"],
        ("x", "y"),
        "coordinate_frame.axes",
        _RegionMetadataError,
    )

    calculation = _require_mapping(entry["calculation"], "calculation", _RegionMetadataError)
    _require_exact_keys(
        calculation,
        {
            "method",
            "weighting",
            "background_value",
            "pixel_coordinate_convention",
            "implementation",
            "algorithm_version",
        },
        "calculation",
        _RegionMetadataError,
    )
    for key, expected in (
        ("method", "center_of_mass"),
        ("weighting", "uniform_label_pixels"),
        ("background_value", 0),
        ("pixel_coordinate_convention", "integer_indices_are_pixel_centers"),
        ("implementation", "harpy.utils.RasterAggregator.center_of_mass"),
    ):
        _require_equal(calculation[key], expected, f"calculation.{key}", _RegionMetadataError)
    algorithm_version = _require_integer(
        calculation["algorithm_version"],
        "calculation.algorithm_version",
        _RegionMetadataError,
        positive=True,
    )

    coverage = _require_mapping(entry["coverage"], "coverage", _RegionMetadataError)
    _require_exact_keys(
        coverage,
        {"scope", "n_obs", "instance_set_digest"},
        "coverage",
        _RegionMetadataError,
    )
    _require_equal(coverage["scope"], "all_rows_for_region", "coverage.scope", _RegionMetadataError)
    n_obs = _require_integer(coverage["n_obs"], "coverage.n_obs", _RegionMetadataError, positive=True)
    digest = _require_nonempty_string(
        coverage["instance_set_digest"],
        "coverage.instance_set_digest",
        _RegionMetadataError,
    )

    source = _require_mapping(entry["source"], "source", _RegionMetadataError)
    _require_exact_keys(source, {"element_path", "dims", "shape", "dtype"}, "source", _RegionMetadataError)
    _require_equal(source["element_path"], f"labels/{region}", "source.element_path", _RegionMetadataError)
    dims = _require_string_sequence(source["dims"], ("y", "x"), "source.dims", _RegionMetadataError)
    shape_values = _require_sequence(source["shape"], "source.shape", _RegionMetadataError)
    if len(shape_values) != 2:
        raise _RegionMetadataError("source.shape must contain exactly two values.")
    shape = tuple(
        _require_integer(size, f"source.shape[{index}]", _RegionMetadataError, positive=True)
        for index, size in enumerate(shape_values)
    )
    dtype_value = _require_nonempty_string(source["dtype"], "source.dtype", _RegionMetadataError)
    try:
        source_dtype = np.dtype(dtype_value)
    except TypeError as exc:
        raise _RegionMetadataError("source.dtype must be a valid NumPy dtype.") from exc
    if source_dtype.kind not in "iu" or dtype_value != source_dtype.name:
        raise _RegionMetadataError("source.dtype must be a normalized integer NumPy dtype name.")

    generated_by_package: str | None = None
    generated_by_version: str | None = None
    generated_at: str | None = None
    if "generated_by" in entry:
        generated_by = _require_mapping(entry["generated_by"], "generated_by", _RegionMetadataError)
        if not set(generated_by).issubset({"package", "version", "generated_at"}):
            raise _RegionMetadataError("generated_by contains unsupported fields.")
        if "package" in generated_by:
            generated_by_package = _require_nonempty_string(
                generated_by["package"], "generated_by.package", _RegionMetadataError
            )
        if "version" in generated_by:
            generated_by_version = _require_nonempty_string(
                generated_by["version"], "generated_by.version", _RegionMetadataError
            )
        if "generated_at" in generated_by:
            generated_at = _require_nonempty_string(
                generated_by["generated_at"], "generated_by.generated_at", _RegionMetadataError
            )

    try:
        source_signature = CanonicalSourceSignature(
            labels_name=region,
            source_scale="scale0",
            dims=dims,
            shape=shape,
            dtype=source_dtype.name,
        )
        return CanonicalRegionMetadata(
            source_signature=source_signature,
            n_obs=n_obs,
            instance_set_digest=digest,
            algorithm_version=algorithm_version,
            generated_by_package=generated_by_package,
            generated_by_version=generated_by_version,
            generated_at=generated_at,
        )
    except (TypeError, ValueError) as exc:
        raise _RegionMetadataError(str(exc)) from exc


def _preserve_valid_other_regions(
    sdata: SpatialData,
    table: AnnData,
    table_metadata: SpatialDataTableMetadata,
    metadata: CanonicalMetadata,
    existing_matrix: np.ndarray,
    *,
    selected_region: str,
    candidate_matrix: np.ndarray,
) -> dict[str, CanonicalRegionMetadata]:
    preserved: dict[str, CanonicalRegionMetadata] = {}
    for region in sorted(metadata.regions):
        if region == selected_region:
            continue
        stored = metadata.regions[region]
        try:
            live_source = build_canonical_source_signature(sdata, region)
            live_binding = build_canonical_region_binding(table, table_metadata, region)
        except (TypeError, ValueError, KeyError):
            continue
        if (
            stored.source_signature != live_source
            or stored.n_obs != live_binding.n_obs
            or stored.instance_set_digest != live_binding.instance_set_digest
        ):
            continue
        if stored.algorithm_version != CANONICAL_ALGORITHM_VERSION:
            continue
        values = existing_matrix[live_binding.row_positions]
        if not np.isfinite(values).all():
            continue
        candidate_matrix[live_binding.row_positions] = values
        preserved[region] = stored
    return preserved


def _assign_canonical_pair_atomically(
    table: AnnData,
    matrix: np.ndarray,
    metadata_storage: dict[str, object],
) -> None:
    matrix_existed = CANONICAL_OBSM_KEY in table.obsm
    previous_matrix = table.obsm.get(CANONICAL_OBSM_KEY)
    registry_existed = SPATIAL_COORDINATES_KEY in table.uns
    previous_registry = table.uns.get(SPATIAL_COORDINATES_KEY)

    current_registry = previous_registry
    candidate_registry = dict(current_registry) if isinstance(current_registry, Mapping) else {}
    candidate_registry[CANONICAL_OBSM_KEY] = metadata_storage
    try:
        table.obsm[CANONICAL_OBSM_KEY] = matrix
        table.uns[SPATIAL_COORDINATES_KEY] = candidate_registry
    except BaseException:
        try:
            if matrix_existed:
                table.obsm[CANONICAL_OBSM_KEY] = previous_matrix
            else:
                table.obsm.pop(CANONICAL_OBSM_KEY, None)
            if registry_existed:
                table.uns[SPATIAL_COORDINATES_KEY] = previous_registry
            else:
                table.uns.pop(SPATIAL_COORDINATES_KEY, None)
        except BaseException as rollback_error:  # pragma: no cover - catastrophic mapping failure
            raise RuntimeError(
                "Canonical cache assignment failed and rollback could not restore the prior pair."
            ) from rollback_error
        raise


def _validate_canonical_matrix(value: object, n_obs: int) -> np.ndarray | str:
    if issparse(value):
        return "Canonical matrix must be dense."
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError):
        return "Canonical matrix must be a NumPy-compatible dense array."
    if matrix.shape != (n_obs, 2):
        return f"Canonical matrix must have shape ({n_obs}, 2)."
    if matrix.dtype != np.dtype(np.float64):
        return "Canonical matrix dtype must be float64."
    if np.isinf(matrix).any():
        return "Canonical matrix must not contain infinite values."
    return matrix


def _validate_schema_v1_source(source: CanonicalSourceSignature) -> None:
    if source.dims != ("y", "x"):
        raise ValueError("Canonical metadata schema version 1 requires source dims (`y`, `x`).")
    try:
        dtype = np.dtype(source.dtype)
    except TypeError as exc:
        raise ValueError("Canonical source dtype must be a valid NumPy dtype.") from exc
    if dtype.kind not in "iu" or source.dtype != dtype.name:
        raise ValueError("Canonical source dtype must be a normalized integer NumPy dtype name.")


def _serialize_generated_by(metadata: CanonicalRegionMetadata) -> dict[str, str] | None:
    generated_by: dict[str, str] = {}
    if metadata.generated_by_package is not None:
        generated_by["package"] = metadata.generated_by_package
    if metadata.generated_by_version is not None:
        generated_by["version"] = metadata.generated_by_version
    if metadata.generated_at is not None:
        generated_by["generated_at"] = metadata.generated_at
    return generated_by or None


def _report(
    state: CanonicalCacheState,
    labels_name: str,
    metadata: CanonicalMetadata | None,
    source_signature: CanonicalSourceSignature,
    binding: CanonicalRegionBinding,
    *mismatches: CanonicalCacheMismatch,
) -> CanonicalCacheReport:
    return CanonicalCacheReport(
        state=state,
        selected_region=labels_name,
        metadata=metadata,
        source_signature=source_signature,
        binding=binding,
        mismatches=tuple(mismatches),
    )


def _pair_mismatch(code: CanonicalMismatchCode, detail: str | None = None) -> CanonicalCacheMismatch:
    return CanonicalCacheMismatch(code=code, detail=_bounded_detail(detail))


def _region_mismatch(
    code: CanonicalMismatchCode,
    region: str,
    detail: str | None = None,
) -> CanonicalCacheMismatch:
    return CanonicalCacheMismatch(code=code, region=region, detail=_bounded_detail(detail))


def _bounded_detail(detail: str | None) -> str | None:
    if detail is None:
        return None
    return detail[:240]


def _require_mapping(
    value: object,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise error_type(f"{field} must be a mapping.")
    return value


def _require_sequence(
    value: object,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> list[object]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise error_type(f"{field} must be a one-dimensional sequence.")
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise error_type(f"{field} must be a sequence.")


def _require_string_sequence(
    value: object,
    expected: tuple[str, ...],
    field: str,
    error_type: type[ValueError] = _TopLevelContractError,
) -> tuple[str, ...]:
    values = _require_sequence(value, field, error_type)
    if not all(isinstance(item, str) for item in values) or tuple(values) != expected:
        raise error_type(f"{field} must equal {list(expected)!r}.")
    return expected


def _require_exact_keys(
    mapping: Mapping[str, object],
    expected: set[str],
    field: str,
    error_type: type[ValueError],
) -> None:
    if set(mapping) != expected:
        _raise_key_error(mapping, expected, expected, field, error_type)


def _raise_key_error(
    mapping: Mapping[str, object],
    required: set[str],
    allowed: set[str],
    field: str,
    error_type: type[ValueError],
) -> NoReturn:
    missing = sorted(required - set(mapping))
    extra = sorted(set(mapping) - allowed)
    parts: list[str] = []
    if missing:
        parts.append(f"missing {missing!r}")
    if extra:
        parts.append(f"unsupported {extra!r}")
    raise error_type(f"{field} fields are invalid ({'; '.join(parts)}).")


def _require_equal(
    value: object,
    expected: object,
    field: str,
    error_type: type[ValueError] = _TopLevelContractError,
) -> None:
    if isinstance(value, np.generic):
        value = value.item()
    if value != expected:
        raise error_type(f"{field} must equal {expected!r}.")


def _require_integer(
    value: object,
    field: str,
    error_type: type[ValueError] = ValueError,
    *,
    positive: bool = False,
) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise error_type(f"{field} must be an integer.")
    integer = int(value)
    if positive and integer <= 0:
        raise error_type(f"{field} must be a positive integer.")
    return integer


def _require_nonempty_string(
    value: object,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str) or not value:
        raise error_type(f"{field} must be a non-empty string.")
    return value
