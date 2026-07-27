from __future__ import annotations

from spatialdata import SpatialData

from napari_harpy._app_state import HarpyAppState, TableStateChangedEvent
from napari_harpy.core.persistence import TableComponentPath
from napari_harpy.widgets.persistence.controls import TablePersistenceControls


def _dirty_user_class(
    app_state: HarpyAppState,
    sdata: SpatialData,
    *,
    table_name: str = "table",
) -> None:
    app_state.record_table_mutation(
        TableStateChangedEvent(
            sdata=sdata,
            table_name=table_name,
            paths=frozenset({TableComponentPath("obs", ("user_class",))}),
            regions=("blobs_labels",),
            change_kind="updated",
            source="test",
        )
    )


def test_bound_controls_synchronize_after_one_control_writes(qtbot, backed_sdata_blobs: SpatialData) -> None:
    app_state = HarpyAppState()
    first = TablePersistenceControls(app_state)
    second = TablePersistenceControls(app_state)
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.bind(backed_sdata_blobs, "table", "blobs_labels")
    second.bind(backed_sdata_blobs, "table", "blobs_labels")

    _dirty_user_class(app_state, backed_sdata_blobs)

    assert first.write_button.isEnabled()
    assert second.write_button.isEnabled()

    first.write_button.click()

    assert app_state.is_table_dirty(backed_sdata_blobs, "table") is False
    assert not first.write_button.isEnabled()
    assert not second.write_button.isEnabled()


def test_dirty_event_refreshes_only_controls_bound_to_affected_table(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
    sdata_blobs_multi_region: SpatialData,
) -> None:
    app_state = HarpyAppState()
    affected = TablePersistenceControls(app_state)
    unrelated = TablePersistenceControls(app_state)
    qtbot.addWidget(affected)
    qtbot.addWidget(unrelated)
    affected.bind(sdata_blobs, "table", "blobs_labels")
    unrelated.bind(sdata_blobs_multi_region, "table_multi", "blobs_labels")
    refresh_calls: list[str] = []
    monkeypatch.setattr(affected, "refresh", lambda: refresh_calls.append("affected"))
    monkeypatch.setattr(unrelated, "refresh", lambda: refresh_calls.append("unrelated"))

    _dirty_user_class(app_state, sdata_blobs)

    assert refresh_calls == ["affected"]
