from __future__ import annotations

from collections.abc import Callable

import pytest

from napari_harpy.viewer.labels_async import sync_labels_display_after_colormap_change


class _RecordingSignal:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[], None]] = []

    @property
    def connect_count(self) -> int:
        return len(self._callbacks)

    def connect(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)

    def emit(self) -> None:
        for callback in tuple(self._callbacks):
            callback()


class _RecordingSlicingState:
    def __init__(self) -> None:
        self.loaded_data = _RecordingSignal()


class _RecordingEvents:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def colormap(self) -> None:
        self._calls.append("events.colormap")

    def set_data(self) -> None:
        self._calls.append("events.set_data")


class _RecordingLabelsLayer:
    def __init__(self, *, loaded: bool = True) -> None:
        self._loaded = loaded
        self.calls: list[str] = []
        self.events = _RecordingEvents(self.calls)
        self._slicing_state = _RecordingSlicingState()
        self.set_view_slice_call_count = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    def set_loaded(self, loaded: bool) -> None:
        if self._loaded == loaded:
            return
        self._loaded = loaded
        self._slicing_state.loaded_data.emit()

    def set_view_slice(self) -> None:
        self.calls.append("set_view_slice")
        self.set_view_slice_call_count += 1


def _expected_sync_calls() -> list[str]:
    return ["set_view_slice", "events.colormap", "events.set_data"]


def test_sync_labels_display_after_colormap_change_recomputes_current_slice_when_loaded() -> None:
    layer = _RecordingLabelsLayer()

    sync_labels_display_after_colormap_change(layer)  # type: ignore[arg-type]

    assert layer.calls == _expected_sync_calls()
    assert layer.set_view_slice_call_count == 1
    assert layer._slicing_state.loaded_data.connect_count == 0


def test_sync_labels_display_after_colormap_change_fails_for_unsupported_layer() -> None:
    with pytest.raises(AttributeError):
        sync_labels_display_after_colormap_change(object())  # type: ignore[arg-type]


def test_sync_labels_display_after_colormap_change_defers_until_unloaded_layer_is_loaded(qtbot) -> None:
    layer = _RecordingLabelsLayer(loaded=False)

    sync_labels_display_after_colormap_change(layer)  # type: ignore[arg-type]

    assert layer.calls == []
    assert layer.set_view_slice_call_count == 0
    assert layer._slicing_state.loaded_data.connect_count == 1

    layer.set_loaded(True)

    qtbot.waitUntil(lambda: layer.calls == _expected_sync_calls(), timeout=1000)
    assert layer.set_view_slice_call_count == 1


def test_sync_labels_display_after_colormap_change_coalesces_repeated_unloaded_requests(qtbot) -> None:
    layer = _RecordingLabelsLayer(loaded=False)

    sync_labels_display_after_colormap_change(layer)  # type: ignore[arg-type]
    sync_labels_display_after_colormap_change(layer)  # type: ignore[arg-type]
    sync_labels_display_after_colormap_change(layer)  # type: ignore[arg-type]

    assert layer.calls == []
    assert layer._slicing_state.loaded_data.connect_count == 1

    layer.set_loaded(True)

    qtbot.waitUntil(lambda: layer.calls == _expected_sync_calls(), timeout=1000)
    assert layer.set_view_slice_call_count == 1


def test_sync_labels_display_after_colormap_change_rechecks_loaded_before_queued_sync(qtbot) -> None:
    layer = _RecordingLabelsLayer(loaded=False)

    sync_labels_display_after_colormap_change(layer)  # type: ignore[arg-type]
    layer.set_loaded(True)
    layer.set_loaded(False)

    qtbot.wait(20)

    assert layer.calls == []
    assert layer.set_view_slice_call_count == 0

    layer.set_loaded(True)

    qtbot.waitUntil(lambda: layer.calls == _expected_sync_calls(), timeout=1000)
    assert layer.set_view_slice_call_count == 1


def test_sync_labels_display_after_colormap_change_loaded_request_clears_pending_queued_sync(qtbot) -> None:
    layer = _RecordingLabelsLayer(loaded=False)

    sync_labels_display_after_colormap_change(layer)  # type: ignore[arg-type]
    layer.set_loaded(True)
    sync_labels_display_after_colormap_change(layer)  # type: ignore[arg-type]

    assert layer.calls == _expected_sync_calls()
    assert layer.set_view_slice_call_count == 1

    qtbot.wait(20)

    assert layer.calls == _expected_sync_calls()
    assert layer.set_view_slice_call_count == 1
