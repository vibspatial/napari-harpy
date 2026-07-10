from __future__ import annotations

import pytest

from napari_harpy.viewer.labels_async import sync_labels_display_after_colormap_change


class _RecordingEvents:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def colormap(self) -> None:
        self._calls.append("events.colormap")

    def set_data(self) -> None:
        self._calls.append("events.set_data")


class _RecordingLabelsLayer:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.events = _RecordingEvents(self.calls)
        self.set_view_slice_call_count = 0

    def set_view_slice(self) -> None:
        self.calls.append("set_view_slice")
        self.set_view_slice_call_count += 1


def test_sync_labels_display_after_colormap_change_recomputes_current_slice() -> None:
    layer = _RecordingLabelsLayer()

    sync_labels_display_after_colormap_change(layer)  # type: ignore[arg-type]

    assert layer.calls == ["set_view_slice", "events.colormap", "events.set_data"]
    assert layer.set_view_slice_call_count == 1


def test_sync_labels_display_after_colormap_change_fails_for_unsupported_layer() -> None:
    with pytest.raises(AttributeError):
        sync_labels_display_after_colormap_change(object())  # type: ignore[arg-type]
