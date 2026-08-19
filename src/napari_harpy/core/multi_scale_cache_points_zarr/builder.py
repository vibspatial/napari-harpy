"""Coordinate guarded construction and publication of one Zarr cache generation."""

from __future__ import annotations

import shutil
import uuid
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from filelock import FileLock, Timeout

from napari_harpy.core.multi_scale_cache_points.models import ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points.validation import _require_parquet_source_unchanged
from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import _plan_points_cache
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import _CatalogWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.models import _INT64_MAX, _require_integer_in_range
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.writer.bridge import (
    _BridgeWriterConfig,
    _write_bridge_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.catalog import _write_staged_cache_catalog
from napari_harpy.core.multi_scale_cache_points_zarr.writer.exact import (
    _ExactWriterConfig,
    _write_exact_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.spatial import (
    _SpatialWriterConfig,
    _write_spatial_levels,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.staging_validation import _validate_staged_cache

_COMPLETION_MARKER = "COMPLETED"
_LOCK_SUFFIX = ".build-lock"


def _default_zarr_write_settings() -> _ZarrWriteSettings:
    """Return the default physical chunk, shard, and codec settings."""
    return _ZarrWriteSettings(
        point_chunk_rows=4_096,
        point_shard_rows=131_072,
        range_chunk_rows=8_192,
        range_shard_rows=131_072,
        codec_id="zstd-v1",
    )


@dataclass(frozen=True)
class _PointsCacheBuilderConfig:
    """Configure one complete Zarr cache build.

    Parameters
    ----------
    leaf_tile_size
        Exact and Bridge logical tile edge in intrinsic source coordinates.
    overview_point_budget
        Maximum planned representative count in the terminal overview level.
    dask_worker_count
        Positive local threaded-worker count used by Exact construction.
    zarr_settings
        One physical chunk, shard, and codec profile shared by every level.
    catalog_settings
        Physical chunk and shard settings for catalog arrays.
    max_open_exact_readers
        Optional Bridge bound for retained Exact bucket-reader metadata.
        ``None`` retains every Exact reader for the duration of Bridge writing.
    max_open_finer_readers
        Optional Spatial bound for retained immediate-finer bucket readers.
        ``None`` retains every input reader during each Spatial level.
    """

    leaf_tile_size: int
    overview_point_budget: int
    dask_worker_count: int
    zarr_settings: _ZarrWriteSettings = field(default_factory=_default_zarr_write_settings)
    catalog_settings: _CatalogWriteSettings = field(default_factory=_CatalogWriteSettings)
    max_open_exact_readers: int | None = None
    max_open_finer_readers: int | None = None

    def __post_init__(self) -> None:
        _require_integer_in_range(self.leaf_tile_size, "leaf_tile_size", minimum=1, maximum=_INT64_MAX)
        _require_integer_in_range(
            self.overview_point_budget,
            "overview_point_budget",
            minimum=1,
            maximum=_INT64_MAX,
        )
        _require_integer_in_range(
            self.dask_worker_count,
            "dask_worker_count",
            minimum=1,
            maximum=_INT64_MAX,
        )
        if not isinstance(self.zarr_settings, _ZarrWriteSettings):
            raise ValueError("`zarr_settings` must be _ZarrWriteSettings.")
        if not isinstance(self.catalog_settings, _CatalogWriteSettings):
            raise ValueError("`catalog_settings` must be _CatalogWriteSettings.")
        for name in ("max_open_exact_readers", "max_open_finer_readers"):
            value = getattr(self, name)
            if value is not None:
                _require_integer_in_range(value, name, minimum=1, maximum=_INT64_MAX)


@contextmanager
def _acquire_output_build_lock(path: Path) -> Generator[None]:
    """Hold non-blocking inter-process ownership of one publication path.

    The generation UUID identifies a staged cache and makes its staging path
    unique. This sibling lock independently prevents participating builders
    from concurrently installing, replacing, or restoring the same final
    output, during a first build as well as replacement.

    ``FileLock`` uses a platform-aware inter-process lock and releases active
    ownership when the process exits. The coordination pathname may remain on
    disk after release, so its presence is not evidence of an active builder;
    only a failed non-blocking acquisition establishes contention.
    """
    lock = FileLock(path, timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise FileExistsError(f"Another cache builder currently owns the output lock: {path}.") from error
    try:
        yield
    finally:
        lock.release()


def _build_points_cache_zarr(
    validated: ValidatedPointsSource,
    *,
    output_path: Path,
    temporary_directory_root: Path,
    config: _PointsCacheBuilderConfig,
) -> Path:
    """Build, independently validate, and publish one complete Zarr cache.

    Parameters
    ----------
    validated
        Canonical content-validated Parquet points source.
    output_path
        Final local cache directory; its parent must already exist.
    temporary_directory_root
        Existing caller-owned directory for disposable Dask shuffle storage.
    config
        Logical planning, execution, Zarr storage, and catalog settings.

    The builder owns one unique sibling staging generation and the publication
    lock, but never owns the caller's temporary root or canonical Parquet
    source. No public output is changed until all levels, catalog metadata, and
    independent staged validation have completed and a final metadata-only
    source guard succeeds.
    """
    if not output_path.parent.is_dir():
        raise ValueError("`output_path.parent` must be an existing directory.")
    if output_path.is_symlink():
        raise ValueError("`output_path` must not be a symbolic link.")
    if not temporary_directory_root.is_dir():
        raise ValueError("`temporary_directory_root` must be an existing directory.")

    output_resolved = output_path.resolve(strict=False)
    temporary_resolved = temporary_directory_root.resolve(strict=True)
    if (
        output_resolved == temporary_resolved
        or output_resolved in temporary_resolved.parents
        or temporary_resolved in output_resolved.parents
    ):
        raise ValueError("Cache output and temporary roots must be separate directory trees.")

    cache_generation_id = str(uuid.uuid4())
    lock_path = output_path.with_name(f"{output_path.name}{_LOCK_SUFFIX}")

    with _acquire_output_build_lock(lock_path):
        existing_generation_id = _preflight_existing_output(output_path)
        _require_parquet_source_unchanged(validated)
        plan = _plan_points_cache(
            validated,
            leaf_tile_size=config.leaf_tile_size,
            overview_point_budget=config.overview_point_budget,
        )
        staging_root = output_path.with_name(f"{output_path.name}.staging-{cache_generation_id}")
        staging_root.mkdir()

        build_error: Exception | None = None
        try:
            exact_result = _write_exact_level(
                validated,
                plan,
                staging_root=staging_root,
                temporary_directory_root=temporary_directory_root,
                config=_ExactWriterConfig(
                    zarr_settings=config.zarr_settings,
                    dask_worker_count=config.dask_worker_count,
                ),
            )
            if len(plan.levels) == 1:
                level_results = (exact_result,)
            else:
                bridge_result = _write_bridge_level(
                    exact_result,
                    plan,
                    staging_root=staging_root,
                    config=_BridgeWriterConfig(
                        zarr_settings=config.zarr_settings,
                        max_open_exact_readers=config.max_open_exact_readers,
                    ),
                )
                spatial_results = _write_spatial_levels(
                    bridge_result,
                    plan,
                    staging_root=staging_root,
                    config=_SpatialWriterConfig(
                        zarr_settings=config.zarr_settings,
                        max_open_finer_readers=config.max_open_finer_readers,
                    ),
                )
                level_results = (exact_result, bridge_result, *spatial_results)

            _write_staged_cache_catalog(
                validated,
                plan,
                level_results,
                staging_root=staging_root,
                cache_generation_id=cache_generation_id,
                settings=config.catalog_settings,
            )
            del exact_result, level_results, plan
            _validate_staged_cache(staging_root)
            _require_parquet_source_unchanged(validated)
            _write_completion_marker(staging_root, cache_generation_id=cache_generation_id)
            return _publish_staged_generation(
                staging_root,
                output_path,
                expected_existing_generation_id=existing_generation_id,
            )
        except Exception as error:
            build_error = error
            raise
        finally:
            if staging_root.exists():
                try:
                    _remove_owned_staging(staging_root)
                except OSError as cleanup_error:
                    if build_error is None:
                        raise
                    raise ExceptionGroup(
                        "Cache construction and staging cleanup both failed.",
                        (build_error, cleanup_error),
                    ) from None
def _preflight_existing_output(output_path: Path) -> str | None:
    """Return the generation ID of a replaceable output, or ``None`` if absent."""
    if output_path.is_symlink():
        raise ValueError("Cache output path must not be a symbolic link.")
    if not output_path.exists():
        return None
    if not output_path.is_dir():
        raise ValueError("Cache output path exists but is not a directory.")
    return _require_completed_generation(output_path)


def _require_completed_generation(cache_root: Path) -> str:
    """Validate a completion marker and the reopened root/catalog layouts."""
    marker_path = cache_root / _COMPLETION_MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError(f"Cache generation `{cache_root}` has no valid `{_COMPLETION_MARKER}` marker.")
    try:
        marker_bytes = marker_path.read_bytes()
        marker_text = marker_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"Cache generation `{cache_root}` has an unreadable completion marker.") from error
    if not marker_text.endswith("\n") or marker_text.count("\n") != 1:
        raise ValueError("Cache completion marker must contain exactly one canonical UUID plus newline.")
    marker_generation_id = marker_text[:-1]
    try:
        parsed = uuid.UUID(marker_generation_id)
    except (ValueError, AttributeError) as error:
        raise ValueError("Cache completion marker does not contain a canonical UUID.") from error
    if str(parsed) != marker_generation_id:
        raise ValueError("Cache completion marker does not contain a canonical lowercase UUID.")

    with _CatalogReader(cache_root) as reader:
        root_generation_id = reader.attributes.cache_generation_id
    if marker_generation_id != root_generation_id:
        raise ValueError("Cache completion marker UUID does not match the root cache-generation UUID.")
    return marker_generation_id


def _write_completion_marker(staging_root: Path, *, cache_generation_id: str) -> None:
    """Write the final staged-generation mutation with exclusive semantics."""
    marker_path = staging_root / _COMPLETION_MARKER
    with marker_path.open("x", encoding="utf-8", newline="") as marker:
        marker.write(f"{cache_generation_id}\n")


def _publish_staged_generation(
    staging_root: Path,
    output_path: Path,
    *,
    expected_existing_generation_id: str | None,
) -> Path:
    """Install one completed staging tree and restore the old tree on failure."""
    _require_completed_generation(staging_root)
    observed_existing_generation_id = _preflight_existing_output(output_path)
    if observed_existing_generation_id != expected_existing_generation_id:
        raise RuntimeError("Cache output changed after builder preflight; refusing publication.")

    backup_path: Path | None = None
    if observed_existing_generation_id is not None:
        backup_path = _unique_sibling_path(output_path, label="backup")
        output_path.rename(backup_path)
    try:
        staging_root.rename(output_path)
    except OSError as install_error:
        if backup_path is not None and backup_path.exists() and not output_path.exists():
            try:
                backup_path.rename(output_path)
            except OSError as rollback_error:
                raise ExceptionGroup(
                    "Installing the staged cache and restoring the previous generation both failed.",
                    (install_error, rollback_error),
                ) from None
        raise

    if backup_path is not None:
        try:
            shutil.rmtree(backup_path)
        except OSError as cleanup_error:
            warnings.warn(
                f"Published cache successfully but could not remove completed backup `{backup_path}`: "
                f"{cleanup_error}",
                RuntimeWarning,
                stacklevel=2,
            )
    return output_path


def _unique_sibling_path(path: Path, *, label: str) -> Path:
    """Return an absent UUID-named sibling without creating it."""
    while True:
        candidate = path.with_name(f"{path.name}.{label}-{uuid.uuid4()}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate


def _remove_owned_staging(staging_root: Path) -> None:
    """Remove only the exact unique staging directory owned by this build."""
    if staging_root.is_symlink() or not staging_root.is_dir():
        raise OSError(f"Owned staging path is no longer a removable directory: {staging_root}.")
    shutil.rmtree(staging_root)
