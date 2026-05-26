from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_harpy.widgets.viewer.status_card import (
    _ViewerStatusCardSpec,
    build_image_loaded_card_spec,
    build_points_layer_card_spec,
    build_primary_labels_loaded_card_spec,
    build_primary_shapes_loaded_card_spec,
    build_styled_labels_card_spec,
    build_styled_shapes_card_spec,
    build_viewer_error_card_spec,
    build_viewer_feedback_card_spec,
)


def test_viewer_status_card_spec_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="Invalid status card kind"):
        _ViewerStatusCardSpec(
            title="Debug",
            lines=("Unexpected card kind.",),
            kind="debug",
        )


def test_build_viewer_error_card_spec() -> None:
    spec = build_viewer_error_card_spec("Image Load Error", ["Overlay mode requires a channel."])

    assert spec.title == "Image Load Error"
    assert spec.lines == ("Overlay mode requires a channel.",)
    assert spec.kind == "error"
    assert spec.tooltip_message is None


def test_build_viewer_feedback_card_spec_defaults_error_title_from_legacy_flag() -> None:
    spec = build_viewer_feedback_card_spec("Old error", is_error=True)

    assert spec.title == "Viewer Error"
    assert spec.lines == ("Old error",)
    assert spec.kind == "error"


def test_build_points_layer_card_spec_reports_warnings() -> None:
    request = SimpleNamespace(
        identity=SimpleNamespace(points_name="cells", coordinate_system="global"),
        selection=SimpleNamespace(
            index_column="cell_type",
            loaded_count=1250,
            warning="Only rendered a sampled subset.",
        ),
    )
    result = SimpleNamespace(
        created=True,
        selected_value_count=200,
        categorical_limit=102,
        categorical_coloring_disabled=True,
    )

    spec = build_points_layer_card_spec(request, result)

    assert spec.title == "Points Layer Created With Warning"
    assert spec.kind == "warning"
    assert spec.lines[0] == "Created points layer for `cells` by `cell_type` with 1,250 point(s)."
    assert "Only rendered a sampled subset." in spec.lines
    assert "Categorical coloring is disabled for 200 selected values" in spec.lines[-1]


def test_build_points_layer_card_spec_uses_tooltip_for_shortened_names() -> None:
    points_name = "points_" + "x" * 80
    request = SimpleNamespace(
        identity=SimpleNamespace(points_name=points_name, coordinate_system="global"),
        selection=SimpleNamespace(
            index_column="gene",
            loaded_count=12,
            warning=None,
        ),
    )
    result = SimpleNamespace(
        created=False,
        selected_value_count=1,
        categorical_limit=102,
        categorical_coloring_disabled=False,
    )

    spec = build_points_layer_card_spec(request, result)

    assert spec.title == "Points Layer Updated"
    assert points_name not in spec.lines[0]
    assert spec.tooltip_message == f"Updated points layer for `{points_name}` by `gene` with 12 point(s)."


def test_build_primary_labels_loaded_card_spec_uses_tooltip_for_shortened_names() -> None:
    labels_name = "labels_" + "x" * 80
    request = SimpleNamespace(labels_name=labels_name)
    result = SimpleNamespace(
        created=True,
        value_kind=None,
        palette_source=None,
        coercion_applied=False,
    )

    spec = build_primary_labels_loaded_card_spec(request, result)

    assert spec.title == "Labels Layer Created"
    assert spec.kind == "success"
    assert labels_name not in spec.lines[0]
    assert spec.tooltip_message == f"Created labels layer for `{labels_name}`."


def test_build_styled_labels_card_spec_reports_stored_palette() -> None:
    request = SimpleNamespace(
        labels_name="blobs_labels",
        selected_color_source=SimpleNamespace(source_kind="obs_column", value_key="cell_type"),
    )
    result = SimpleNamespace(
        created=True,
        value_kind="categorical",
        palette_source="stored",
        coercion_applied=False,
    )

    spec = build_styled_labels_card_spec(request, result)

    assert spec.title == "Colored Overlay Created"
    assert spec.kind == "success"
    assert spec.lines == (
        'Created colored overlay for obs["cell_type"] on labels element `blobs_labels`.',
        "Used the stored categorical palette.",
    )


def test_build_styled_labels_card_spec_uses_tooltip_for_shortened_names() -> None:
    labels_name = "labels_" + "x" * 80
    request = SimpleNamespace(
        labels_name=labels_name,
        selected_color_source=SimpleNamespace(source_kind="obs_column", value_key="cell_type"),
    )
    result = SimpleNamespace(
        created=True,
        value_kind="categorical",
        palette_source="stored",
        coercion_applied=False,
    )

    spec = build_styled_labels_card_spec(request, result)

    assert labels_name not in spec.lines[0]
    assert spec.tooltip_message == f'Created colored overlay for obs["cell_type"] on labels element `{labels_name}`.'


def test_build_styled_labels_card_spec_reports_instance_coloring() -> None:
    request = SimpleNamespace(
        labels_name="blobs_labels",
        selected_color_source=SimpleNamespace(source_kind="obs_column", value_key="instance_id"),
    )
    result = SimpleNamespace(
        created=True,
        value_kind="instance",
        palette_source=None,
        coercion_applied=False,
    )

    spec = build_styled_labels_card_spec(request, result)

    assert spec.title == "Colored Overlay Created"
    assert spec.kind == "success"
    assert spec.lines[-1] == "Used instance label colors."


def test_build_styled_labels_card_spec_rejects_missing_color_source() -> None:
    request = SimpleNamespace(
        labels_name="blobs_labels",
        selected_color_source=None,
    )
    result = SimpleNamespace(
        created=True,
        value_kind="categorical",
        palette_source="stored",
        coercion_applied=False,
    )

    with pytest.raises(ValueError, match="Styled labels status requires a selected color source"):
        build_styled_labels_card_spec(request, result)


def test_build_primary_shapes_loaded_card_spec_reports_skipped_geometries() -> None:
    request = SimpleNamespace(shapes_name="blobs_circles")
    result = SimpleNamespace(
        created=True,
        value_kind=None,
        palette_source=None,
        coercion_applied=False,
        skipped_geometry_count=2,
    )

    spec = build_primary_shapes_loaded_card_spec(request, result)

    assert spec.title == "Shapes Layer Created With Warning"
    assert spec.kind == "warning"
    assert spec.lines == (
        "Created shapes layer for `blobs_circles`.",
        "Skipped 2 empty, invalid, or unsupported geometries while loading renderable shapes.",
    )


def test_build_styled_shapes_card_spec_combines_palette_and_geometry_warnings() -> None:
    request = SimpleNamespace(
        shapes_name="blobs_circles",
        selected_color_source=SimpleNamespace(value_key="cell_type"),
    )
    result = SimpleNamespace(
        created=False,
        value_kind="categorical",
        palette_source="default_missing",
        coercion_applied=False,
        skipped_geometry_count=1,
    )

    spec = build_styled_shapes_card_spec(request, result)

    assert spec.title == "Styled Shapes Updated With Warning"
    assert spec.kind == "warning"
    assert spec.lines == (
        'Updated styled shapes layer for column "cell_type" on shapes element `blobs_circles`.',
        "Used the default categorical palette because no stored palette was present.",
        "Skipped 1 empty, invalid, or unsupported geometries while loading renderable shapes.",
    )


def test_build_styled_shapes_card_spec_uses_tooltip_for_shortened_names() -> None:
    shapes_name = "shapes_" + "x" * 80
    request = SimpleNamespace(
        shapes_name=shapes_name,
        selected_color_source=SimpleNamespace(value_key="cell_type"),
    )
    result = SimpleNamespace(
        created=True,
        value_kind="continuous",
        palette_source=None,
        coercion_applied=False,
        skipped_geometry_count=0,
    )

    spec = build_styled_shapes_card_spec(request, result)

    assert shapes_name not in spec.lines[0]
    assert (
        spec.tooltip_message == f'Created styled shapes layer for column "cell_type" on shapes element `{shapes_name}`.'
    )


def test_build_styled_shapes_card_spec_rejects_missing_color_source() -> None:
    request = SimpleNamespace(
        shapes_name="blobs_circles",
        selected_color_source=None,
    )
    result = SimpleNamespace(
        created=True,
        value_kind="categorical",
        palette_source="stored",
        coercion_applied=False,
        skipped_geometry_count=0,
    )

    with pytest.raises(ValueError, match="Styled shapes status requires a selected color source"):
        build_styled_shapes_card_spec(request, result)


def test_build_image_loaded_card_spec_reports_overlay_channels() -> None:
    request = SimpleNamespace(image_name="blobs_image")
    result = SimpleNamespace(
        layers=(object(), object()),
        mode="overlay",
        created=True,
        channels=(0, 2),
        channel_names=("DAPI", "CD8"),
    )

    spec = build_image_loaded_card_spec(request, result)

    assert spec.title == "Image Layer Created"
    assert spec.kind == "success"
    assert spec.lines == ("Created image overlay for `blobs_image` with channels `DAPI`, `CD8`.",)


def test_build_image_loaded_card_spec_falls_back_to_overlay_channel_indices() -> None:
    request = SimpleNamespace(image_name="blobs_image")
    result = SimpleNamespace(
        layers=(object(),),
        mode="overlay",
        created=True,
        channels=(0,),
    )

    spec = build_image_loaded_card_spec(request, result)

    assert spec.lines == ("Created image overlay for `blobs_image` with channel `0`.",)
