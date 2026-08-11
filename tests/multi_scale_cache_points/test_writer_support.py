from __future__ import annotations

import numpy as np

from napari_harpy.core.multi_scale_cache_points.build_plan import _LevelBuildPlan, _LevelKind
from napari_harpy.core.multi_scale_cache_points.writer.support import (
    BUCKET_HASH_METHOD,
    _bucket_count_for_level,
    _tile_bucket_ids,
)


def test_splitmix64_bucket_mapping_has_fixed_vectors() -> None:
    tile_x = np.array([0, 1, 0, 1, 2**32 - 1], dtype=np.uint32)
    tile_y = np.array([0, 0, 1, 1, 2**32 - 1], dtype=np.uint32)

    assert BUCKET_HASH_METHOD == "harpy-tile-splitmix64-v1"
    assert _tile_bucket_ids(tile_x, tile_y, bucket_count=69).tolist() == [16, 26, 43, 1, 2]


def test_bucket_count_targets_two_million_rows() -> None:
    level = _LevelBuildPlan(
        level=0,
        kind=_LevelKind.EXACT,
        tile_size=512,
        grid_width=1,
        grid_height=1,
        max_points_per_tile=None,
        point_count_upper_bound=136_578_750,
    )

    assert _bucket_count_for_level(level) == 69
