from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import napari_harpy.core.multi_scale_cache_points_zarr.writer.bridge as bridge_module
from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
    _LevelBuildPlan,
    _LevelKind,
    _PointsCacheBuildPlan,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.sampling import _select_sampled_tile_indices
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import _validate_bucket
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_writer import _BucketWriter
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketPlan,
    _BucketWriteResult,
    _LevelWriteResult,
    _PlannedTile,
    _ZarrWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.bridge import (
    _assign_bridge_buckets,
    _BridgeWriterConfig,
    _write_bridge_level,
)


def _settings() -> _ZarrWriteSettings:
    return _ZarrWriteSettings(2, 4, 2, 4, "zstd-v1")


def _plan(*, capacity: int = 4) -> _PointsCacheBuildPlan:
    exact = _LevelBuildPlan(
        level=0,
        kind=_LevelKind.EXACT,
        tile_size=10,
        grid_width=2,
        grid_height=2,
        max_points_per_tile=None,
        point_count_upper_bound=15,
    )
    bridge = _LevelBuildPlan(
        level=1,
        kind=_LevelKind.BRIDGE,
        tile_size=10,
        grid_width=2,
        grid_height=2,
        max_points_per_tile=capacity,
        point_count_upper_bound=15,
    )
    return _PointsCacheBuildPlan(
        x_origin=0.0,
        y_origin=0.0,
        leaf_tile_size=10,
        overview_point_budget=15,
        levels=(exact, bridge),
    )


def _payload(point_id: np.ndarray, *, value_variant: int = 0) -> _PointPayload:
    row = np.arange(len(point_id), dtype=np.float32)
    return _PointPayload(
        x_rel=np.ascontiguousarray((row * 3) % 10, dtype=np.float32),
        y_rel=np.ascontiguousarray((row * 7) % 10, dtype=np.float32),
        value_id=np.ascontiguousarray((point_id * 5 + value_variant) % 7, dtype=np.uint32),
        point_id=np.ascontiguousarray(point_id, dtype=np.uint64),
    )


def _write_exact_fixture(staging_root: Path, *, value_variant: int = 0) -> _LevelWriteResult:
    staging_root.mkdir()
    # Two physical Exact buckets deliberately distribute globally adjacent tiles
    # across different input stores. Every descriptor still represents one
    # complete logical tile.
    bucket_tiles = {
        0: (
            (_PlannedTile(0, 0, 3), _payload(np.array([2, 0, 1], dtype=np.uint64), value_variant=value_variant)),
            (
                _PlannedTile(0, 1, 5),
                _payload(np.array([14, 10, 13, 11, 12], dtype=np.uint64), value_variant=value_variant),
            ),
        ),
        1: (
            (
                _PlannedTile(1, 0, 7),
                _payload(np.array([9, 3, 8, 4, 7, 5, 6], dtype=np.uint64), value_variant=value_variant),
            ),
        ),
    }
    results: list[_BucketWriteResult] = []
    for bucket_id, tiles in bucket_tiles.items():
        plan = _BucketPlan(
            level=0,
            bucket_id=bucket_id,
            tiles=tuple(tile for tile, _ in tiles),
            settings=_settings(),
        )
        with _BucketWriter(staging_root, plan) as writer:
            for tile, payload in tiles:
                writer.write_tile(tile.tile_x, tile.tile_y, payload)
            results.append(writer.finalize())
    return _LevelWriteResult(buckets=tuple(results))


def _payloads_by_tile(result: _LevelWriteResult, staging_root: Path) -> dict[tuple[int, int], _PointPayload]:
    payloads: dict[tuple[int, int], _PointPayload] = {}
    readers: dict[int, _BucketReader] = {}
    try:
        for descriptor in result.tile_descriptors:
            reader = readers.get(descriptor.bucket_id)
            if reader is None:
                reader = _BucketReader(staging_root, level=descriptor.level, bucket_id=descriptor.bucket_id)
                readers[descriptor.bucket_id] = reader.__enter__()
            payloads[(descriptor.tile_x, descriptor.tile_y)] = reader.read_construction_payload(descriptor)
    finally:
        for reader in readers.values():
            reader.__exit__(None, None, None)
    return payloads


def _rows_by_point_id(payload: _PointPayload) -> dict[int, tuple[float, float, int]]:
    return {
        int(point_id): (float(x_rel), float(y_rel), int(value_id))
        for x_rel, y_rel, value_id, point_id in zip(
            payload.x_rel,
            payload.y_rel,
            payload.value_id,
            payload.point_id,
            strict=True,
        )
    }


def test_bridge_writer_reads_exact_zarr_and_persists_deterministic_subsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    exact_result = _write_exact_fixture(staging)
    plan = _plan()
    # Exercise multiple output buckets despite the tiny fixture. Tile hashing
    # leaves destination IDs nonempty at both zero and one.
    monkeypatch.setattr(bridge_module, "_bucket_count_for_level", lambda _level, **_kwargs: 2)

    result = _write_bridge_level(
        exact_result,
        plan,
        staging_root=staging,
        config=_BridgeWriterConfig(_settings(), max_open_exact_readers=1),
    )

    assert result.level == 1
    assert [bucket.bucket_id for bucket in result.buckets] == [0, 1]
    assert [(tile.tile_x, tile.tile_y, tile.n_points) for tile in result.tile_descriptors] == [
        (0, 0, 3),
        (1, 0, 4),
        (0, 1, 4),
    ]
    assert result.point_count == 11
    assert not list((staging / "levels/level_1").rglob("*.parquet"))

    exact_payloads = _payloads_by_tile(exact_result, staging)
    bridge_payloads = _payloads_by_tile(result, staging)
    for coordinate, exact_payload in exact_payloads.items():
        bridge_payload = bridge_payloads[coordinate]
        selected = _select_sampled_tile_indices(
            exact_payload.x_rel,
            exact_payload.y_rel,
            exact_payload.point_id,
            level=1,
            tile_x=coordinate[0],
            tile_y=coordinate[1],
            tile_size=10,
            target=4,
        )
        assert set(bridge_payload.point_id.tolist()) == set(exact_payload.point_id[selected].tolist())
        exact_rows = _rows_by_point_id(exact_payload)
        assert _rows_by_point_id(bridge_payload) == {
            point_id: exact_rows[point_id] for point_id in bridge_payload.point_id.tolist()
        }

    for bucket in result.buckets:
        assert _validate_bucket(staging, level=1, bucket_id=bucket.bucket_id) == bucket


def test_bridge_membership_does_not_depend_on_exact_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_module, "_bucket_count_for_level", lambda _level, **_kwargs: 1)
    first_staging = tmp_path / "first"
    changed_staging = tmp_path / "changed"
    first = _write_bridge_level(
        _write_exact_fixture(first_staging),
        _plan(),
        staging_root=first_staging,
        config=_BridgeWriterConfig(_settings(), max_open_exact_readers=2),
    )
    changed = _write_bridge_level(
        _write_exact_fixture(changed_staging, value_variant=3),
        _plan(),
        staging_root=changed_staging,
        config=_BridgeWriterConfig(_settings(), max_open_exact_readers=2),
    )

    first_payloads = _payloads_by_tile(first, first_staging)
    changed_payloads = _payloads_by_tile(changed, changed_staging)
    assert {coordinate: sorted(payload.point_id.tolist()) for coordinate, payload in first_payloads.items()} == {
        coordinate: sorted(payload.point_id.tolist()) for coordinate, payload in changed_payloads.items()
    }


def test_bridge_default_reader_capacity_keeps_every_exact_bucket_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    exact_result = _write_exact_fixture(staging)
    observed_capacities: list[int] = []
    reader_cache = bridge_module._BucketReaderCache

    def recording_reader_cache(cache_root: Path, *, max_open_readers: int) -> object:
        observed_capacities.append(max_open_readers)
        return reader_cache(cache_root, max_open_readers=max_open_readers)

    monkeypatch.setattr(bridge_module, "_BucketReaderCache", recording_reader_cache)
    monkeypatch.setattr(bridge_module, "_bucket_count_for_level", lambda _level, **_kwargs: 1)

    _write_bridge_level(
        exact_result,
        _plan(),
        staging_root=staging,
        config=_BridgeWriterConfig(_settings()),
    )

    assert observed_capacities == [exact_result.bucket_count]


def test_bridge_routing_omits_empty_bucket_ids_and_orders_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptors = (
        _TileDescriptor(0, 0, 0, 1, 0, 2),
        _TileDescriptor(0, 1, 0, 0, 1, 2),
        _TileDescriptor(0, 2, 0, 0, 0, 2),
    )
    monkeypatch.setattr(
        bridge_module,
        "_tile_bucket_ids",
        lambda _tile_x, _tile_y, *, bucket_count: np.array([2, 0, 2], dtype=np.uint64),
    )

    grouped = _assign_bridge_buckets(descriptors, bucket_count=3)

    assert list(grouped) == [2, 0]
    assert [(tile.tile_x, tile.tile_y) for tile in grouped[2]] == [(0, 0), (1, 0)]
    assert [(tile.tile_x, tile.tile_y) for tile in grouped[0]] == [(0, 1)]
    assert 1 not in grouped


def test_bridge_rejects_missing_plan_or_preexisting_output(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    exact_result = _write_exact_fixture(staging)
    exact_only = replace(_plan(), levels=(_plan().levels[0],), overview_point_budget=15)
    config = _BridgeWriterConfig(_settings(), max_open_exact_readers=1)

    with pytest.raises(ValueError, match="no Bridge level"):
        _write_bridge_level(exact_result, exact_only, staging_root=staging, config=config)

    (staging / "levels/level_1").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _write_bridge_level(exact_result, _plan(), staging_root=staging, config=config)


def test_bridge_failure_removes_current_partial_bucket_but_preserves_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    exact_result = _write_exact_fixture(staging)
    monkeypatch.setattr(bridge_module, "_bucket_count_for_level", lambda _level, **_kwargs: 1)
    monkeypatch.setattr(
        bridge_module,
        "_select_sampled_tile_indices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected sampling failure")),
    )

    with pytest.raises(RuntimeError, match="injected sampling failure"):
        _write_bridge_level(
            exact_result,
            _plan(),
            staging_root=staging,
            config=_BridgeWriterConfig(_settings(), max_open_exact_readers=1),
        )

    assert (staging / "levels/level_0/bucket-000.zarr").is_dir()
    assert (staging / "levels/level_0/bucket-001.zarr").is_dir()
    assert (staging / "levels/level_1").is_dir()
    assert not list((staging / "levels/level_1").iterdir())


@pytest.mark.parametrize("failure", ["read", "write"])
def test_bridge_injected_io_failure_closes_and_removes_current_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    staging = tmp_path / failure
    exact_result = _write_exact_fixture(staging)
    monkeypatch.setattr(bridge_module, "_bucket_count_for_level", lambda _level, **_kwargs: 1)
    if failure == "read":
        monkeypatch.setattr(
            _BucketReader,
            "read_construction_payload",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected read failure")),
        )
    else:
        monkeypatch.setattr(
            _BucketWriter,
            "write_tile",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected write failure")),
        )

    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        _write_bridge_level(
            exact_result,
            _plan(),
            staging_root=staging,
            config=_BridgeWriterConfig(_settings(), max_open_exact_readers=1),
        )

    assert not list((staging / "levels/level_1").iterdir())
    assert (staging / "levels/level_0/bucket-000.zarr").is_dir()


def test_bridge_config_rejects_invalid_settings_and_reader_bound() -> None:
    with pytest.raises(ValueError, match="zarr_settings"):
        _BridgeWriterConfig(object(), max_open_exact_readers=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_open_exact_readers"):
        _BridgeWriterConfig(_settings(), max_open_exact_readers=0)
