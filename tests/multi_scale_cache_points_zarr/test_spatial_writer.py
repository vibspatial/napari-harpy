from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import napari_harpy.core.multi_scale_cache_points_zarr.writer.spatial as spatial_module
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
from napari_harpy.core.multi_scale_cache_points_zarr.writer.spatial import (
    _assign_spatial_buckets,
    _CoarserTileInput,
    _group_finer_descriptors,
    _rebase_finer_coordinates,
    _SpatialWriterConfig,
    _write_spatial_levels,
)

_BRIDGE_TILE_IDS = {
    (0, 0): (0, 1),
    (1, 0): (2, 3),
    (2, 0): (4,),
    (0, 1): (5, 6),
    (1, 1): (7, 8),
    (2, 2): (9, 10),
}


def _settings() -> _ZarrWriteSettings:
    return _ZarrWriteSettings(2, 4, 2, 4, "zstd-v1")


def _level(
    level: int,
    kind: _LevelKind,
    *,
    tile_size: int,
    grid_width: int,
    grid_height: int,
    capacity: int | None,
    upper_bound: int,
) -> _LevelBuildPlan:
    return _LevelBuildPlan(
        level=level,
        kind=kind,
        tile_size=tile_size,
        grid_width=grid_width,
        grid_height=grid_height,
        max_points_per_tile=capacity,
        point_count_upper_bound=upper_bound,
    )


def _spatial_plan() -> _PointsCacheBuildPlan:
    exact = _level(
        0,
        _LevelKind.EXACT,
        tile_size=10,
        grid_width=3,
        grid_height=3,
        capacity=None,
        upper_bound=14,
    )
    bridge = _level(
        1,
        _LevelKind.BRIDGE,
        tile_size=10,
        grid_width=3,
        grid_height=3,
        capacity=4,
        upper_bound=14,
    )
    first_spatial = _level(
        2,
        _LevelKind.SPATIAL,
        tile_size=20,
        grid_width=2,
        grid_height=2,
        capacity=5,
        upper_bound=10,
    )
    overview = _level(
        3,
        _LevelKind.SPATIAL,
        tile_size=40,
        grid_width=1,
        grid_height=1,
        capacity=5,
        upper_bound=5,
    )
    return _PointsCacheBuildPlan(
        x_origin=0.0,
        y_origin=0.0,
        leaf_tile_size=10,
        overview_point_budget=5,
        levels=(exact, bridge, first_spatial, overview),
    )


def _terminal_bridge_plan() -> _PointsCacheBuildPlan:
    spatial = _spatial_plan()
    return _PointsCacheBuildPlan(
        x_origin=0.0,
        y_origin=0.0,
        leaf_tile_size=10,
        overview_point_budget=14,
        levels=spatial.levels[:2],
    )


def _payload(point_ids: tuple[int, ...]) -> _PointPayload:
    ids = np.asarray(point_ids, dtype=np.uint64)
    return _PointPayload(
        x_rel=np.ascontiguousarray((ids * 3) % 9 + 0.25, dtype=np.float32),
        y_rel=np.ascontiguousarray((ids * 5) % 9 + 0.5, dtype=np.float32),
        value_id=np.ascontiguousarray(ids % 3, dtype=np.uint32),
        point_id=ids,
    )


def _write_bridge_fixture(staging_root: Path) -> _LevelWriteResult:
    staging_root.mkdir()
    bucket_coordinates = {
        0: ((0, 0), (2, 0), (1, 1), (2, 2)),
        1: ((1, 0), (0, 1)),
    }
    results: list[_BucketWriteResult] = []
    for bucket_id, coordinates in bucket_coordinates.items():
        ordered_coordinates = tuple(sorted(coordinates, key=lambda coordinate: (coordinate[1], coordinate[0])))
        bucket_plan = _BucketPlan(
            level=1,
            bucket_id=bucket_id,
            tiles=tuple(
                _PlannedTile(tile_x, tile_y, len(_BRIDGE_TILE_IDS[(tile_x, tile_y)]))
                for tile_x, tile_y in ordered_coordinates
            ),
            settings=_settings(),
        )
        with _BucketWriter(staging_root, bucket_plan) as writer:
            for tile_x, tile_y in ordered_coordinates:
                writer.write_tile(tile_x, tile_y, _payload(_BRIDGE_TILE_IDS[(tile_x, tile_y)]))
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


def _expected_coarser_candidates(
    finer_payloads: dict[tuple[int, int], _PointPayload],
    *,
    coarser_tile_x: int,
    coarser_tile_y: int,
    finer_tile_size: int,
) -> _PointPayload:
    contributors = sorted(
        (
            (coordinate, payload)
            for coordinate, payload in finer_payloads.items()
            if coordinate[0] // 2 == coarser_tile_x and coordinate[1] // 2 == coarser_tile_y
        ),
        key=lambda item: (item[0][1], item[0][0]),
    )
    return _PointPayload(
        x_rel=np.ascontiguousarray(
            np.concatenate(
                [
                    payload.x_rel + np.float32((coordinate[0] - 2 * coarser_tile_x) * finer_tile_size)
                    for coordinate, payload in contributors
                ]
            ),
            dtype=np.float32,
        ),
        y_rel=np.ascontiguousarray(
            np.concatenate(
                [
                    payload.y_rel + np.float32((coordinate[1] - 2 * coarser_tile_y) * finer_tile_size)
                    for coordinate, payload in contributors
                ]
            ),
            dtype=np.float32,
        ),
        value_id=np.ascontiguousarray(
            np.concatenate([payload.value_id for _, payload in contributors]), dtype=np.uint32
        ),
        point_id=np.ascontiguousarray(
            np.concatenate([payload.point_id for _, payload in contributors]), dtype=np.uint64
        ),
    )


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


@pytest.mark.parametrize(
    ("quadrant", "expected_x", "expected_y"),
    [
        ((0, 0), [0.0, 10.0], [1.0, 9.0]),
        ((1, 0), [10.0, 20.0], [1.0, 9.0]),
        ((0, 1), [0.0, 10.0], [11.0, 19.0]),
        ((1, 1), [10.0, 20.0], [11.0, 19.0]),
    ],
)
def test_rebase_finer_coordinates_covers_every_quadrant_and_closed_edge(
    quadrant: tuple[int, int],
    expected_x: list[float],
    expected_y: list[float],
) -> None:
    payload = _PointPayload(
        x_rel=np.array([0, 10], dtype=np.float32),
        y_rel=np.array([1, 9], dtype=np.float32),
        value_id=np.array([0, 1], dtype=np.uint32),
        point_id=np.array([0, 1], dtype=np.uint64),
    )

    x_rel, y_rel = _rebase_finer_coordinates(
        payload,
        finer_tile_x=4 + quadrant[0],
        finer_tile_y=6 + quadrant[1],
        coarser_tile_x=2,
        coarser_tile_y=3,
        finer_tile_size=10,
    )

    assert x_rel.tolist() == expected_x
    assert y_rel.tolist() == expected_y
    assert x_rel.dtype == np.float32 and x_rel.flags.c_contiguous
    assert y_rel.dtype == np.float32 and y_rel.flags.c_contiguous


def test_rebase_rejects_wrong_coarser_tile_and_out_of_tile_coordinates() -> None:
    valid = _payload((0,))
    with pytest.raises(ValueError, match="quadrant"):
        _rebase_finer_coordinates(
            valid,
            finer_tile_x=4,
            finer_tile_y=4,
            coarser_tile_x=0,
            coarser_tile_y=0,
            finer_tile_size=10,
        )

    invalid = _PointPayload(
        x_rel=np.array([11], dtype=np.float32),
        y_rel=np.array([1], dtype=np.float32),
        value_id=np.array([0], dtype=np.uint32),
        point_id=np.array([0], dtype=np.uint64),
    )
    with pytest.raises(ValueError, match="inside the finer tile"):
        _rebase_finer_coordinates(
            invalid,
            finer_tile_x=0,
            finer_tile_y=0,
            coarser_tile_x=0,
            coarser_tile_y=0,
            finer_tile_size=10,
        )


def test_coarser_tile_grouping_handles_four_contributors_and_sparse_edges() -> None:
    plan = _spatial_plan()
    descriptors = (
        _TileDescriptor(1, 0, 0, 0, 0, 2),
        _TileDescriptor(1, 1, 0, 1, 0, 2),
        _TileDescriptor(1, 1, 1, 0, 1, 2),
        _TileDescriptor(1, 0, 1, 1, 1, 2),
        _TileDescriptor(1, 0, 2, 2, 2, 1),
    )

    grouped = _group_finer_descriptors(
        descriptors,
        finer_level=plan.levels[1],
        coarser_level=plan.levels[2],
    )

    assert [(tile.tile_x, tile.tile_y, tile.candidate_count) for tile in grouped] == [(0, 0, 8), (1, 1, 1)]
    assert [(descriptor.tile_x, descriptor.tile_y) for descriptor in grouped[0].finer_descriptors] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]

    with pytest.raises(ValueError, match="unique"):
        _CoarserTileInput(0, 0, (descriptors[0], descriptors[0]))


@pytest.mark.parametrize("contributor_count", [1, 2, 3, 4])
def test_coarser_tile_grouping_accepts_every_nonempty_quadrant_count(contributor_count: int) -> None:
    plan = _spatial_plan()
    coordinates = ((0, 0), (1, 0), (0, 1), (1, 1))[:contributor_count]
    descriptors = tuple(
        _TileDescriptor(1, 0, index, tile_x, tile_y, index + 1) for index, (tile_x, tile_y) in enumerate(coordinates)
    )

    grouped = _group_finer_descriptors(
        descriptors,
        finer_level=plan.levels[1],
        coarser_level=plan.levels[2],
    )

    assert len(grouped) == 1
    assert grouped[0].candidate_count == sum(range(1, contributor_count + 1))
    assert grouped[0].finer_descriptors == descriptors


def test_spatial_routing_omits_empty_destinations_and_orders_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptors = (
        _TileDescriptor(1, 0, 0, 0, 0, 1),
        _TileDescriptor(1, 0, 1, 2, 0, 1),
        _TileDescriptor(1, 0, 2, 0, 2, 1),
    )
    tiles = (
        _CoarserTileInput(0, 0, (descriptors[0],)),
        _CoarserTileInput(1, 0, (descriptors[1],)),
        _CoarserTileInput(0, 1, (descriptors[2],)),
    )
    monkeypatch.setattr(
        spatial_module,
        "_tile_bucket_ids",
        lambda _tile_x, _tile_y, *, bucket_count: np.array([2, 0, 2], dtype=np.uint64),
    )

    grouped = _assign_spatial_buckets(tiles, bucket_count=3)

    assert list(grouped) == [2, 0]
    assert [(tile.tile_x, tile.tile_y) for tile in grouped[2]] == [(0, 0), (0, 1)]
    assert [(tile.tile_x, tile.tile_y) for tile in grouped[0]] == [(1, 0)]
    assert 1 not in grouped


def test_spatial_writer_builds_nested_multilevel_zarr_pyramid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    bridge_result = _write_bridge_fixture(staging)
    plan = _spatial_plan()
    monkeypatch.setattr(
        spatial_module,
        "_bucket_count_for_level",
        lambda level: 2 if level.level == 2 else 1,
    )
    observed_reader_caches: list[tuple[int, object]] = []
    reader_cache = spatial_module._BucketReaderCache

    def recording_reader_cache(cache_root: Path, *, max_open_readers: int) -> object:
        cache = reader_cache(cache_root, max_open_readers=max_open_readers)
        observed_reader_caches.append((max_open_readers, cache))
        return cache

    monkeypatch.setattr(spatial_module, "_BucketReaderCache", recording_reader_cache)

    results = _write_spatial_levels(
        bridge_result,
        plan,
        staging_root=staging,
        config=_SpatialWriterConfig(_settings()),
    )

    assert [result.level for result in results] == [2, 3]
    assert [result.point_count for result in results] == [8, 5]
    assert [(tile.tile_x, tile.tile_y, tile.n_points) for tile in results[0].tile_descriptors] == [
        (0, 0, 5),
        (1, 0, 1),
        (1, 1, 2),
    ]
    assert [(tile.tile_x, tile.tile_y, tile.n_points) for tile in results[1].tile_descriptors] == [(0, 0, 5)]
    assert [capacity for capacity, _ in observed_reader_caches] == [bridge_result.bucket_count, results[0].bucket_count]
    assert all(cache.open_reader_count == 0 for _, cache in observed_reader_caches)  # type: ignore[attr-defined]

    finer_result = bridge_result
    for result, finer_level, coarser_level in zip(results, plan.levels[1:-1], plan.levels[2:], strict=True):
        finer_payloads = _payloads_by_tile(finer_result, staging)
        output_payloads = _payloads_by_tile(result, staging)
        capacity = coarser_level.max_points_per_tile
        assert capacity is not None
        for coordinate, output in output_payloads.items():
            candidates = _expected_coarser_candidates(
                finer_payloads,
                coarser_tile_x=coordinate[0],
                coarser_tile_y=coordinate[1],
                finer_tile_size=finer_level.tile_size,
            )
            selected = _select_sampled_tile_indices(
                candidates.x_rel,
                candidates.y_rel,
                candidates.point_id,
                level=coarser_level.level,
                tile_x=coordinate[0],
                tile_y=coordinate[1],
                tile_size=coarser_level.tile_size,
                target=capacity,
            )
            expected_rows = _rows_by_point_id(candidates.take(selected))
            assert _rows_by_point_id(output) == expected_rows
        for bucket in result.buckets:
            assert _validate_bucket(staging, level=result.level, bucket_id=bucket.bucket_id) == bucket
        finer_result = result

    assert not list((staging / "levels").rglob("*.parquet"))


def test_terminal_bridge_returns_no_spatial_results(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    bridge_result = _write_bridge_fixture(staging)

    assert (
        _write_spatial_levels(
            bridge_result,
            _terminal_bridge_plan(),
            staging_root=staging,
            config=_SpatialWriterConfig(_settings()),
        )
        == ()
    )
    assert not (staging / "levels/level_2").exists()


def test_spatial_writer_applies_explicit_reader_bound_to_each_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    bridge_result = _write_bridge_fixture(staging)
    monkeypatch.setattr(spatial_module, "_bucket_count_for_level", lambda _level: 1)
    observed_capacities: list[int] = []
    reader_cache = spatial_module._BucketReaderCache

    def recording_reader_cache(cache_root: Path, *, max_open_readers: int) -> object:
        observed_capacities.append(max_open_readers)
        return reader_cache(cache_root, max_open_readers=max_open_readers)

    monkeypatch.setattr(spatial_module, "_BucketReaderCache", recording_reader_cache)

    _write_spatial_levels(
        bridge_result,
        _spatial_plan(),
        staging_root=staging,
        config=_SpatialWriterConfig(_settings(), max_open_finer_readers=1),
    )

    assert observed_capacities == [1, 1]


@pytest.mark.parametrize("failure", ["read", "rebase", "sample", "write"])
def test_spatial_failure_removes_active_bucket_and_preserves_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    staging = tmp_path / failure
    bridge_result = _write_bridge_fixture(staging)
    monkeypatch.setattr(spatial_module, "_bucket_count_for_level", lambda _level: 1)
    if failure == "read":
        monkeypatch.setattr(
            _BucketReader,
            "read_construction_payload",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected read failure")),
        )
    elif failure == "rebase":
        monkeypatch.setattr(
            spatial_module,
            "_rebase_finer_coordinates",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected rebase failure")),
        )
    elif failure == "sample":
        monkeypatch.setattr(
            spatial_module,
            "_select_sampled_tile_indices",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected sample failure")),
        )
    else:
        monkeypatch.setattr(
            _BucketWriter,
            "write_tile",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected write failure")),
        )

    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        _write_spatial_levels(
            bridge_result,
            _spatial_plan(),
            staging_root=staging,
            config=_SpatialWriterConfig(_settings(), max_open_finer_readers=1),
        )

    assert (staging / "levels/level_1/bucket-000.zarr").is_dir()
    assert (staging / "levels/level_1/bucket-001.zarr").is_dir()
    assert (staging / "levels/level_2").is_dir()
    assert not list((staging / "levels/level_2").iterdir())


def test_spatial_later_level_failure_preserves_completed_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    bridge_result = _write_bridge_fixture(staging)
    monkeypatch.setattr(spatial_module, "_bucket_count_for_level", lambda _level: 1)
    sample = spatial_module._select_sampled_tile_indices

    def fail_terminal(*args: object, **kwargs: object) -> np.ndarray:
        if kwargs["level"] == 3:
            raise RuntimeError("injected terminal failure")
        return sample(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(spatial_module, "_select_sampled_tile_indices", fail_terminal)

    with pytest.raises(RuntimeError, match="injected terminal failure"):
        _write_spatial_levels(
            bridge_result,
            _spatial_plan(),
            staging_root=staging,
            config=_SpatialWriterConfig(_settings()),
        )

    assert list((staging / "levels/level_2").glob("bucket-*.zarr"))
    assert (staging / "levels/level_3").is_dir()
    assert not list((staging / "levels/level_3").iterdir())
    assert (staging / "levels/level_1/bucket-000.zarr").is_dir()


def test_spatial_rejects_preexisting_output_and_invalid_config(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    bridge_result = _write_bridge_fixture(staging)
    (staging / "levels/level_2").mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        _write_spatial_levels(
            bridge_result,
            _spatial_plan(),
            staging_root=staging,
            config=_SpatialWriterConfig(_settings()),
        )
    with pytest.raises(ValueError, match="zarr_settings"):
        _SpatialWriterConfig(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_open_finer_readers"):
        _SpatialWriterConfig(_settings(), max_open_finer_readers=0)
