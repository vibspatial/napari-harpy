from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np

from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import _validate_bucket
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_writer import _BucketWriter
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketPlan,
    _PlannedTile,
    _ZarrWriteSettings,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Characterize the standalone sharded Zarr bucket primitive.")
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _payload(point_count: int, *, point_id_start: int, value_count: int) -> _PointPayload:
    point_id = np.arange(point_id_start, point_id_start + point_count, dtype=np.uint64)
    # Unequal deterministic run lengths exercise both localized values and
    # selected values whose runs intersect several inner chunks.
    value_id = np.floor(
        np.linspace(0, value_count, point_count, endpoint=False, dtype=np.float64)
    ).astype(np.uint32)
    x_rel = np.remainder(point_id, np.uint64(512)).astype(np.float32)
    y_rel = np.remainder(point_id // np.uint64(512), np.uint64(512)).astype(np.float32)
    return _PointPayload(x_rel=x_rel, y_rel=y_rel, value_id=value_id, point_id=point_id)


def _directory_summary(root: Path) -> tuple[int, int]:
    file_count = 0
    byte_count = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            file_count += 1
            byte_count += (Path(directory) / filename).stat().st_size
    return file_count, byte_count


def _time_read(operation: Callable[[], object], *, repeats: int = 5) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        timings.append(perf_counter() - start)
    return timings


def main() -> None:
    """Build, validate, and time representative synthetic bucket operations."""
    args = _parse_args()
    settings = _ZarrWriteSettings(
        point_chunk_rows=4_096,
        point_shard_rows=131_072,
        range_chunk_rows=8_192,
        range_shard_rows=131_072,
        codec_id="zstd-v1",
    )
    scenarios = (
        ("average_exact", 18_700, 512),
        ("dense_exact", 108_598, 4_096),
        ("bridge", 4_096, 256),
    )
    tiles = tuple(_PlannedTile(index, 0, point_count) for index, (_, point_count, _) in enumerate(scenarios))
    plan = _BucketPlan(level=0, bucket_id=0, tiles=tiles, settings=settings)

    temporary_parent = None if args.work_directory is None else str(args.work_directory)
    if args.work_directory is not None:
        args.work_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="napari-harpy-zarr-bucket-", dir=temporary_parent) as workspace_text:
        workspace = Path(workspace_text)
        point_id_start = 0
        write_start = perf_counter()
        with _BucketWriter(workspace, plan) as writer:
            for tile, (_, point_count, value_count) in zip(tiles, scenarios, strict=True):
                writer.write_tile(
                    tile.tile_x,
                    tile.tile_y,
                    _payload(point_count, point_id_start=point_id_start, value_count=value_count),
                )
                point_id_start += point_count
            result = writer.finalize()
        write_seconds = perf_counter() - write_start

        validation_start = perf_counter()
        validated = _validate_bucket(workspace, level=0, bucket_id=0)
        validation_seconds = perf_counter() - validation_start
        if validated != result:
            raise RuntimeError("Independent bucket validation reconstructed a different result.")

        file_count, byte_count = _directory_summary(workspace / result.bucket_path)
        reads: dict[str, object] = {}
        with _BucketReader(workspace, level=0, bucket_id=0) as reader:
            for descriptor, (name, _, value_count) in zip(result.tile_descriptors, scenarios, strict=True):
                complete = _time_read(lambda descriptor=descriptor: reader.read_construction_payload(descriptor))
                localized = np.array([value_count // 2], dtype=np.uint32)
                distributed = np.array([0, value_count // 2, value_count - 1], dtype=np.uint32)
                localized_times = _time_read(
                    lambda descriptor=descriptor, selected=localized: reader.read_display_payload(descriptor, selected)
                )
                distributed_times = _time_read(
                    lambda descriptor=descriptor, selected=distributed: reader.read_display_payload(
                        descriptor, selected
                    )
                )
                reads[name] = {
                    "complete_seconds": complete,
                    "localized_one_value_seconds": localized_times,
                    "distributed_three_values_seconds": distributed_times,
                }

        report = {
            "schema_version": "harpy-zarr-bucket-characterization-v1",
            "configuration": {
                "point_chunk_rows": settings.point_chunk_rows,
                "point_shard_rows": settings.point_shard_rows,
                "range_chunk_rows": settings.range_chunk_rows,
                "range_shard_rows": settings.range_shard_rows,
                "codec_id": settings.codec_id,
            },
            "bucket": {
                "point_count": result.point_count,
                "range_count": result.range_count,
                "filesystem_file_count": file_count,
                "stored_bytes": byte_count,
                "write_seconds": write_seconds,
                "validation_seconds": validation_seconds,
            },
            "reads": reads,
        }
        output = json.dumps(report, indent=2, sort_keys=True)
        if args.json_output is None:
            print(output)
        else:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(output + "\n")


if __name__ == "__main__":
    main()
