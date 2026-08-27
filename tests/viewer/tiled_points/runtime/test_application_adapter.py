from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import napari_harpy.viewer.adapter as adapter_module
from napari_harpy.core.multi_scale_cache_points_zarr.reader import _read_cache_dataset_info
from napari_harpy.viewer.adapter import TiledPointsLayerBinding, ViewerAdapter
from napari_harpy.viewer.tiled_points.application import (
    TiledPointsApplicationSettings,
    TiledPointsCacheDescriptor,
)


class _Emitter:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self.callbacks.append(callback)

    def emit(self, value: object) -> None:
        event = SimpleNamespace(value=value)
        for callback in tuple(self.callbacks):
            callback(event)


class _Layers(list[object]):
    def __init__(self) -> None:
        super().__init__()
        self.events = SimpleNamespace(inserted=_Emitter(), removed=_Emitter(), reordered=_Emitter())
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)

    def _select_only(self, layer: object) -> None:
        self.selection.active = layer

    def remove(self, layer: object) -> None:
        super().remove(layer)
        self.events.removed.emit(layer)


class _Viewer:
    def __init__(self, timeline: list[str]) -> None:
        self.layers = _Layers()
        self.timeline = timeline

    def add_layer(self, layer: object) -> None:
        self.timeline.append("insert")
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)


class _Runtime:
    def __init__(self, layer, cache_root, settings, *, initial_requested_value_ids=None) -> None:
        del layer, cache_root, settings
        self.initial_requested_value_ids = initial_requested_value_ids
        self.selection_updates: list[tuple[int, ...] | None] = []
        self.close_count = 0

    @property
    def closed(self) -> bool:
        return self.close_count > 0

    def set_selected_value_ids(self, requested_value_ids: tuple[int, ...] | None) -> bool:
        self.selection_updates.append(requested_value_ids)
        return True

    def close(self) -> bool:
        self.close_count += 1
        return self.close_count == 1


def test_adapter_creates_reuses_and_closes_one_persistent_tiled_binding(
    monkeypatch,
    real_cache_root: Path,
) -> None:
    timeline: list[str] = []
    viewer = _Viewer(timeline)
    adapter = ViewerAdapter(viewer)
    sdata = object()
    descriptor = TiledPointsCacheDescriptor(real_cache_root, _read_cache_dataset_info(real_cache_root))

    monkeypatch.setattr(adapter_module, "register_tiled_points_layer", lambda: timeline.append("register"))

    def create_runtime(*args, **kwargs):
        timeline.append("runtime")
        return _Runtime(*args, **kwargs)

    monkeypatch.setattr(adapter_module, "_TiledPointsLayerRuntime", create_runtime)
    monkeypatch.setattr(adapter_module, "_get_points_affine_transform", lambda *args: np.eye(3))

    first = adapter.ensure_tiled_points_layer(
        sdata=sdata,  # type: ignore[arg-type]
        points_name="transcripts",
        coordinate_system="global",
        descriptor=descriptor,
        requested_value_ids=(0,),
        hard_render_point_budget=100_000,
        settings=TiledPointsApplicationSettings(max_cpu_tile_bytes=1_000, max_gpu_tile_bytes=1_000),
    )

    assert first.created
    assert isinstance(first.binding, TiledPointsLayerBinding)
    assert timeline == ["register", "runtime", "insert"]
    runtime = first.binding.runtime
    assert runtime.initial_requested_value_ids == (0,)  # type: ignore[attr-defined]

    first.binding.layer.visible = False
    first.binding.layer.point_diameter = 7.0
    second = adapter.ensure_tiled_points_layer(
        sdata=sdata,  # type: ignore[arg-type]
        points_name="transcripts",
        coordinate_system="global",
        descriptor=descriptor,
        requested_value_ids=(1,),
        hard_render_point_budget=50_000,
        settings=TiledPointsApplicationSettings(max_cpu_tile_bytes=1_000, max_gpu_tile_bytes=1_000),
    )

    assert not second.created
    assert second.binding.layer is first.binding.layer
    assert second.binding.layer.visible is False
    assert second.binding.layer.point_diameter == 7.0
    assert runtime.selection_updates == [(1,)]  # type: ignore[attr-defined]
    assert adapter.activate_layer(second.binding.layer)
    assert viewer.layers.selection.active is second.binding.layer

    viewer.layers.remove(second.binding.layer)
    assert runtime.close_count == 1  # type: ignore[attr-defined]
    assert adapter.layer_bindings.get_binding(second.binding.layer) is None
