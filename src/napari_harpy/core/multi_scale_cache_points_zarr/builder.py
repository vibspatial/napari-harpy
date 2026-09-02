"""Coordinate guarded construction and publication of one Zarr cache generation."""

from __future__ import annotations

import shutil
import uuid
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import zarr
from filelock import FileLock, Timeout
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import _plan_points_cache
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    PUBLICATION_STATE_COMPLETE,
    PUBLICATION_STATE_STAGING,
    _CatalogWriteSettings,
    _parse_cache_attributes,
    _ValueMajorWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.hashing import TARGET_POINTS_PER_BUCKET
from napari_harpy.core.multi_scale_cache_points_zarr.models import _INT64_MAX, _require_integer_in_range
from napari_harpy.core.multi_scale_cache_points_zarr.source.models import ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points_zarr.source.validation import _require_parquet_source_unchanged
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
    target_points_per_bucket
        Positive target used to derive deterministic bucket counts at every
        serialized level. The default preserves the adopted two-million-point
        policy and the configured value is persisted in cache metadata.
    zarr_settings
        One physical chunk, shard, and codec profile shared by every level.
    catalog_settings
        Physical chunk and shard settings for catalog arrays.
    value_major_settings
        Physical layout and bounded construction settings for the mandatory
        all-level value-major location sidecar.
    max_open_value_major_readers
        Optional bound for retained tile-major source-bucket readers during
        value-major construction. ``None`` retains every source reader for the
        active level and is the default.
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
    target_points_per_bucket: int = TARGET_POINTS_PER_BUCKET
    zarr_settings: _ZarrWriteSettings = field(default_factory=_default_zarr_write_settings)
    catalog_settings: _CatalogWriteSettings = field(default_factory=_CatalogWriteSettings)
    value_major_settings: _ValueMajorWriteSettings = field(default_factory=_ValueMajorWriteSettings)
    max_open_value_major_readers: int | None = None
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
        _require_integer_in_range(
            self.target_points_per_bucket,
            "target_points_per_bucket",
            minimum=1,
            maximum=_INT64_MAX,
        )
        if not isinstance(self.zarr_settings, _ZarrWriteSettings):
            raise ValueError("`zarr_settings` must be _ZarrWriteSettings.")
        if not isinstance(self.catalog_settings, _CatalogWriteSettings):
            raise ValueError("`catalog_settings` must be _CatalogWriteSettings.")
        if not isinstance(self.value_major_settings, _ValueMajorWriteSettings):
            raise ValueError("`value_major_settings` must be _ValueMajorWriteSettings.")
        for name in (
            "max_open_value_major_readers",
            "max_open_exact_readers",
            "max_open_finer_readers",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_integer_in_range(value, name, minimum=1, maximum=_INT64_MAX)


@contextmanager
def _acquire_output_build_lock(path: Path) -> Generator[None]:
    """Hold non-blocking inter-process ownership of one publication path.

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
    source. No public output is changed until all levels, catalog metadata,
    mandatory value-major sidecars, and independent staged validation have
    completed and a final metadata-only source guard succeeds.
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

    # The lock path is stable for this final output and serializes participating
    # builders. The UUID instead identifies only this cache generation: it makes
    # the staging path unique and is later persisted in root metadata.
    cache_generation_id = str(uuid.uuid4())
    lock_path = output_path.with_name(f"{output_path.name}{_LOCK_SUFFIX}")
    staging_root = output_path.with_name(f"{output_path.name}.staging-{cache_generation_id}")

    with _acquire_output_build_lock(lock_path):
        existing_generation_id = _get_existing_complete_cache_generation_id(output_path)
        # Reject a validated source whose Parquet metadata changed before this
        # build began. This checks only metadata; it does not reread every point row.
        _require_parquet_source_unchanged(validated)
        plan = _plan_points_cache(
            validated,
            leaf_tile_size=config.leaf_tile_size,
            overview_point_budget=config.overview_point_budget,
        )
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
                    target_points_per_bucket=config.target_points_per_bucket,
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
                        target_points_per_bucket=config.target_points_per_bucket,
                        max_open_exact_readers=config.max_open_exact_readers,
                    ),
                )
                spatial_results = _write_spatial_levels(
                    bridge_result,
                    plan,
                    staging_root=staging_root,
                    config=_SpatialWriterConfig(
                        zarr_settings=config.zarr_settings,
                        target_points_per_bucket=config.target_points_per_bucket,
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
                value_major_settings=config.value_major_settings,
                max_open_value_major_readers=config.max_open_value_major_readers,
                temporary_directory_root=temporary_directory_root,
                target_points_per_bucket=config.target_points_per_bucket,
            )
            del exact_result, level_results, plan
            _validate_staged_cache(staging_root)
            # Construction may be long-running. Check the source again before
            # completion so a cache built while Parquet changed is not published.
            _require_parquet_source_unchanged(validated)
            _mark_cache_generation_complete(staging_root, cache_generation_id=cache_generation_id)
            # Confirm that the output observed before construction is still the
            # generation about to be replaced before any directory is renamed.
            observed_existing_generation_id = _get_existing_complete_cache_generation_id(output_path)
            if observed_existing_generation_id != existing_generation_id:
                raise RuntimeError(
                    "Cache output changed after the builder's initial output check; refusing publication."
                )
            return _publish_staged_generation(staging_root, output_path)
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


def _get_existing_complete_cache_generation_id(output_path: Path) -> str | None:
    """Return the generation ID of a replaceable output, or ``None`` if absent."""
    if output_path.is_symlink():
        raise ValueError("Cache output path must not be a symbolic link.")
    if not output_path.exists():
        return None
    if not output_path.is_dir():
        raise ValueError("Cache output path exists but is not a directory.")
    return _require_complete_cache_generation_id(output_path)


def _require_complete_cache_generation_id(cache_root: Path) -> str:
    """Require complete root metadata and return its cache-generation UUID."""
    with _CatalogReader(cache_root) as reader:
        attributes = reader.attributes
        if attributes.publication_state != PUBLICATION_STATE_COMPLETE:
            raise ValueError("Cache root publication_state is not 'complete'.")
        return attributes.cache_generation_id


def _mark_cache_generation_complete(staging_root: Path, *, cache_generation_id: str) -> None:
    """Make publication state the final mutation of a validated generation."""
    with LocalStore(staging_root, read_only=False) as store:
        root = zarr.open_group(store=store, mode="r+", zarr_format=3, use_consolidated=False)
        attributes = _parse_cache_attributes(dict(root.attrs))
        if attributes.cache_generation_id != cache_generation_id:
            raise ValueError("Staged root cache-generation UUID changed before completion.")
        if attributes.publication_state != PUBLICATION_STATE_STAGING:
            raise ValueError("Only a staging generation can be marked complete.")
        root.update_attributes({"publication_state": PUBLICATION_STATE_COMPLETE})


def _publish_staged_generation(
    staging_root: Path,
    output_path: Path,
) -> Path:
    """Install one completed staging tree and restore the old tree on failure.

    The directory transitions are::

        first build:
            staging(new) -> output(new)

        replacement:
            output(old)  -> backup(old)
            staging(new) -> output(new)
            delete backup(old)

        failed replacement:
            output(old)  -> backup(old)
            staging rename fails
            backup(old)  -> output(old)
    """
    _require_complete_cache_generation_id(staging_root)

    backup_path: Path | None = None
    if output_path.exists():
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
                f"Published cache successfully but could not remove completed backup `{backup_path}`: {cleanup_error}",
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
