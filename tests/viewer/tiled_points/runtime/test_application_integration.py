from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.reader import _read_cache_dataset_info
from napari_harpy.viewer.tiled_points.application import (
    DEFAULT_MAX_CPU_TILE_BYTES,
    DEFAULT_MAX_VERTEX_PAYLOAD_BYTES,
    TiledPointsApplicationSettings,
    TiledPointsCacheDescriptor,
    canonical_value_palette,
)
from napari_harpy.widgets.viewer.tiled_points_controller import _CacheDescriptorJob, _load_cache_descriptor


class _BackedSpatialData:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.points = {"transcripts": object()}

    def is_backed(self) -> bool:
        return True

    def locate_element(self, element: object) -> list[str]:
        assert element is self.points["transcripts"]
        return ["points/transcripts"]


def test_application_settings_freeze_product_residency_defaults() -> None:
    settings = TiledPointsApplicationSettings()

    assert settings.max_bucket_lookup_bytes is None
    assert settings.max_selected_value_index_bytes is None
    assert settings.max_cpu_tile_bytes == DEFAULT_MAX_CPU_TILE_BYTES
    assert settings.max_vertex_payload_bytes == DEFAULT_MAX_VERTEX_PAYLOAD_BYTES
    assert settings.cache_session_settings.max_vertex_payload_bytes == DEFAULT_MAX_VERTEX_PAYLOAD_BYTES


def test_canonical_value_mapping_and_palette_ignore_selection_order(real_cache_root: Path) -> None:
    descriptor = TiledPointsCacheDescriptor(real_cache_root, _read_cache_dataset_info(real_cache_root))

    assert descriptor.requested_value_ids(("B", "A")) == (0, 1)
    assert descriptor.requested_value_ids(("A", "B")) == (0, 1)
    assert descriptor.requested_value_ids("all") is None

    palette = canonical_value_palette(103)
    assert palette.shape == (103, 4)
    assert palette.dtype == np.uint8
    assert np.array_equal(palette[102], palette[0])


def test_nested_descriptor_loading_reads_cache_metadata_without_touching_points(
    tmp_path: Path,
    real_cache_root: Path,
) -> None:
    nested_cache = tmp_path / "points" / "transcripts" / "transcripts_vis_zarr"
    nested_cache.parent.mkdir(parents=True)
    shutil.copytree(real_cache_root, nested_cache)
    sdata = _BackedSpatialData(tmp_path)

    descriptor = _load_cache_descriptor(
        _CacheDescriptorJob(1, sdata, "transcripts", "global", "gene")  # type: ignore[arg-type]
    )

    assert descriptor.cache_root == nested_cache
    assert descriptor.value_names == ("A", "B")
    assert descriptor.dataset_info.value_column == "gene"


def test_nested_descriptor_rejects_a_different_selected_value_column(
    tmp_path: Path,
    real_cache_root: Path,
) -> None:
    nested_cache = tmp_path / "points" / "transcripts" / "transcripts_vis_zarr"
    nested_cache.parent.mkdir(parents=True)
    shutil.copytree(real_cache_root, nested_cache)

    with pytest.raises(ValueError, match="Selected value column 'target'.*cache value column 'gene'"):
        _load_cache_descriptor(
            _CacheDescriptorJob(1, _BackedSpatialData(tmp_path), "transcripts", "global", "target")  # type: ignore[arg-type]
        )
