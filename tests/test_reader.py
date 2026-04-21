from __future__ import annotations

import json
from pathlib import Path

import napari_harpy._reader as reader_module


def _write_spatialdata_zarr_json(path: Path) -> None:
    path.mkdir()
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "attributes": {
                    "spatialdata_attrs": {
                        "version": "0.2",
                        "spatialdata_software_version": "0.7.2",
                    }
                },
                "zarr_format": 3,
            }
        )
    )


def test_get_reader_accepts_spatialdata_zarr_store(tmp_path) -> None:
    store_path = tmp_path / "example.zarr"
    _write_spatialdata_zarr_json(store_path)

    reader = reader_module.get_reader(str(store_path))

    assert callable(reader)


def test_get_reader_rejects_non_spatialdata_zarr_store(tmp_path) -> None:
    store_path = tmp_path / "example.zarr"
    store_path.mkdir()
    (store_path / "zarr.json").write_text(json.dumps({"attributes": {}, "zarr_format": 3}))

    reader = reader_module.get_reader(str(store_path))

    assert reader is None


def test_get_reader_rejects_multiple_paths(tmp_path) -> None:
    first = tmp_path / "a.zarr"
    second = tmp_path / "b.zarr"
    _write_spatialdata_zarr_json(first)
    _write_spatialdata_zarr_json(second)

    reader = reader_module.get_reader([str(first), str(second)])

    assert reader is None


def test_reader_loads_spatialdata_into_harpy_app_state(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "example.zarr"
    _write_spatialdata_zarr_json(store_path)
    fake_sdata = object()
    recorded_sdata: list[object] = []
    requested_viewers: list[object] = []
    window_calls: list[tuple[str, str, bool]] = []
    viewer_dock_raised: list[str] = []

    class _FakeWindow:
        def add_plugin_dock_widget(self, plugin_name: str, widget_name: str, tabify: bool = False) -> tuple[object, object]:
            window_calls.append((plugin_name, widget_name, tabify))
            dock_widget = type("FakeDockWidget", (), {"raise_": lambda self: viewer_dock_raised.append(widget_name)})()
            return dock_widget, object()

    fake_viewer = type("FakeViewer", (), {"window": _FakeWindow()})()

    class _FakeState:
        def set_sdata(self, sdata: object) -> None:
            recorded_sdata.append(sdata)

    monkeypatch.setattr(reader_module, "read_zarr", lambda path: fake_sdata)
    monkeypatch.setattr(reader_module.napari, "current_viewer", lambda: fake_viewer)

    def fake_get_or_create_app_state(viewer: object) -> _FakeState:
        requested_viewers.append(viewer)
        return _FakeState()

    monkeypatch.setattr(reader_module, "get_or_create_app_state", fake_get_or_create_app_state)

    reader = reader_module.get_reader(str(store_path))

    assert reader is not None
    assert reader(str(store_path)) == [(None,)]
    assert requested_viewers == [fake_viewer]
    assert recorded_sdata == [fake_sdata]
    assert window_calls == [
        ("napari-harpy", "Viewer", True),
        ("napari-harpy", "Feature Extraction", True),
        ("napari-harpy", "Object Classification", True),
    ]
    assert viewer_dock_raised == ["Viewer"]


def test_reader_raises_when_no_current_viewer_exists(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "example.zarr"
    _write_spatialdata_zarr_json(store_path)

    monkeypatch.setattr(reader_module, "read_zarr", lambda path: object())
    monkeypatch.setattr(reader_module.napari, "current_viewer", lambda: None)

    reader = reader_module.get_reader(str(store_path))

    assert reader is not None
    try:
        reader(str(store_path))
    except RuntimeError as error:
        assert "requires an active napari viewer" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected the reader to fail when no current napari viewer exists.")
