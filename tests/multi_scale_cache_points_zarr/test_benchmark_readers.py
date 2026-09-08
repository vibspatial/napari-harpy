"""Keep benchmark read paths and instrumentation aligned with reader contracts."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.reader import _IntrinsicViewport, _PointsCacheReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader


def _load_benchmark_module(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load a repository benchmark script by file path without changing sys.path."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark script: {script_path}.")
    module = module_from_spec(spec)
    # Register before execution so dataclasses and explicitly loaded sibling
    # scripts can resolve the module. Pytest restores the entry after the test.
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_cache_to_canvas_timing_hooks_cover_both_physical_routes(
    reader_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _load_benchmark_module("benchmark_tiled_points_cache_to_canvas", monkeypatch)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        index = reader.load_selected_value_index(np.array([0, 1], dtype=np.uint32), max_resident_bytes=None)
        for selected_index, expected_timer in ((None, "bucket_batch"), (index, "value_major_location_read")):
            timings = benchmark._TimingLog()
            with benchmark._TemporaryPatches() as patches:
                benchmark._install_reader_timers(timings, patches)
                plan = reader.plan_viewport(0, _IntrinsicViewport(0, 0, 12, 10), value_index=selected_index)
                result = reader.read_planned_tiles(plan, plan.tile_keys)
            assert result.tiles
            assert timings.calls[expected_timer]


def test_acceptance_tile_timing_labels_diagnostic_filtering(reader_fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = _load_benchmark_module("benchmark_multi_scale_cache_points_zarr_acceptance", monkeypatch)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        result, report = benchmark._time_tile(reader, 0, 0, 0, value_ids=np.array([0], dtype=np.uint32))
    assert result is not None and result.value_id.tolist() == [0, 0]
    assert report["read_mode"] == "complete_tile_major_then_filter"


@pytest.mark.parametrize("fixed_level", [0, None], ids=["fixed-exact", "selected-lod"])
def test_selected_index_benchmark_reads_without_retired_lookup_metrics(
    reader_fixture, fixed_level, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _load_benchmark_module("benchmark_multi_scale_cache_points_zarr_selected_value_index", monkeypatch)

    report = benchmark._measure_selected_viewport(
        reader_fixture.cache_root,
        _IntrinsicViewport(0, 0, 12, 10),
        np.array([0, 1], dtype=np.uint32),
        fixed_level=fixed_level,
        max_resident_bytes=10_000_000,
        point_budget=100_000,
    )
    assert report["first"]["returned_points"] == report["estimated_points"]
    assert report["repeated"]["returned_points"] == report["first"]["returned_points"]


def test_exact_read_measurements_use_complete_arrays_and_reconciled_filter_counts(
    catalog_exact_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Exact benchmark imports this sibling's diagnostic filtering helper.
    _load_benchmark_module("benchmark_multi_scale_cache_points_zarr_bucket", monkeypatch)
    benchmark = _load_benchmark_module("benchmark_multi_scale_cache_points_zarr_exact", monkeypatch)

    fixture = catalog_exact_fixture
    first_descriptor_by_value = {}
    value_tile_counts = [0] * fixture.validated.value_table.num_rows
    for bucket in fixture.result.buckets:
        with _BucketReader(fixture.staging_root, level=0, bucket_id=bucket.bucket_id) as reader:
            for descriptor in bucket.tile_descriptors:
                payload = reader.read_construction_payload(descriptor)
                for value_id in np.unique(payload.value_id):
                    first_descriptor_by_value.setdefault(int(value_id), descriptor)
                    value_tile_counts[int(value_id)] += 1
    report = benchmark._read_measurements(
        fixture.result,
        staging=fixture.staging_root,
        validated=fixture.validated,
        first_descriptor_by_value=first_descriptor_by_value,
        value_tile_counts=value_tile_counts,
    )
    for category in ("common", "median", "rare_localized", "rare_distributed"):
        measurement = report[category]
        assert measurement["read_mode"] == "complete_tile_major_then_filter"
        assert 0 < measurement["logical_rows"] <= measurement["complete_rows_read"]
        assert measurement["decoded_point_rows_per_array"] >= measurement["complete_rows_read"]
        assert measurement["decoded_rows_per_returned_point"] == (
            measurement["decoded_point_rows_per_array"] / measurement["logical_rows"]
        )
        assert measurement["seconds"]
