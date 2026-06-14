from __future__ import annotations

import pytest
from spatialdata import SpatialData

from napari_harpy.core.validation import spatialdata_element_name_exists, validate_new_spatialdata_element_name


def test_validate_new_spatialdata_element_name_rejects_case_variant(sdata_blobs: SpatialData) -> None:
    existing_shapes_name = next(iter(sdata_blobs.shapes))

    with pytest.raises(ValueError, match="already exists"):
        validate_new_spatialdata_element_name(sdata_blobs, existing_shapes_name.upper(), "Shapes element")


def test_validate_new_spatialdata_element_name_rejects_cross_element_collision(sdata_blobs: SpatialData) -> None:
    existing_labels_name = next(iter(sdata_blobs.labels))

    with pytest.raises(ValueError, match="already exists"):
        validate_new_spatialdata_element_name(sdata_blobs, existing_labels_name, "Shapes element")


def test_validate_new_spatialdata_element_name_returns_normalized_name(sdata_blobs: SpatialData) -> None:
    assert validate_new_spatialdata_element_name(sdata_blobs, " new_shapes ", "Shapes element") == "new_shapes"


def test_spatialdata_element_name_exists_matches_case_insensitively_across_element_types(
    sdata_blobs: SpatialData,
) -> None:
    assert spatialdata_element_name_exists(sdata_blobs, next(iter(sdata_blobs.labels)).upper()) is True
    assert spatialdata_element_name_exists(sdata_blobs, "new_shapes") is False
