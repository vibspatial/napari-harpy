from __future__ import annotations

from napari_harpy._widget import HarpyWidget


def test_widget_can_be_instantiated(qtbot) -> None:
    widget = HarpyWidget()

    qtbot.addWidget(widget)

    assert widget is not None
