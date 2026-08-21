from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType

from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
    _INT64_MAX,
    _UINT32_MAX,
    _require_integer_in_range,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader


class _BucketReaderCache:
    """Bound the lifetime of reusable entered Zarr bucket readers.

    Parameters
    ----------
    cache_root
        Cache-generation root containing canonical level bucket paths.
    max_open_readers
        Positive maximum number of entered readers retained at once.

    Notes
    -----
    The least-recently-used reader is closed before a new miss is admitted at
    capacity. Reuse retains initialized Zarr array metadata and explicitly
    loaded immutable lookup indexes; this cache never stores decoded chunks or
    point payloads.
    """

    def __init__(self, cache_root: str | Path, *, max_open_readers: int) -> None:
        _require_integer_in_range(
            max_open_readers,
            "max_open_readers",
            minimum=1,
            maximum=_INT64_MAX,
        )
        self._cache_root = Path(cache_root)
        self._max_open_readers = max_open_readers
        self._readers: OrderedDict[tuple[int, int], _BucketReader] = OrderedDict()
        self._entered = False
        self._open = False

    def __enter__(self) -> _BucketReaderCache:
        if self._entered:
            raise RuntimeError("A bucket reader cache can be entered only once.")
        self._entered = True
        self._open = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self._close_all()
        return False

    def get(self, *, level: int, bucket_id: int) -> _BucketReader:
        """Return one entered reader and mark it most recently used."""
        self._require_open()
        _require_integer_in_range(level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(bucket_id, "bucket_id", maximum=_UINT32_MAX)
        key = (level, bucket_id)
        cached = self._readers.get(key)
        if cached is not None:
            self._readers.move_to_end(key)
            return cached

        if len(self._readers) == self._max_open_readers:
            _, least_recently_used = self._readers.popitem(last=False)
            least_recently_used.__exit__(None, None, None)

        reader = _BucketReader(self._cache_root, level=level, bucket_id=bucket_id)
        # `_BucketReader.__enter__` cleans up its own partially opened state. Add
        # it to the LRU only after the complete strict open succeeds.
        entered = reader.__enter__()
        self._readers[key] = entered
        return entered

    @property
    def open_reader_count(self) -> int:
        """Return the current number of entered readers."""
        return len(self._readers)

    @property
    def loaded_lookup_index_count(self) -> int:
        """Return the number of readers retaining bucket lookup metadata."""
        return sum(reader.lookup_index_loaded for reader in self._readers.values())

    @property
    def resident_lookup_bytes(self) -> int:
        """Return bytes retained by all loaded bucket lookup indexes."""
        return sum(reader.resident_lookup_bytes for reader in self._readers.values())

    def release_lookup_indexes(self, keys: tuple[tuple[int, int], ...]) -> None:
        """Release lookup buffers for the stated already opened buckets."""
        self._require_open()
        for key in keys:
            reader = self._readers.get(key)
            if reader is not None:
                reader.release_lookup_index()

    def _require_open(self) -> None:
        if not self._open:
            raise RuntimeError("Bucket reader cache is not open.")

    def _close_all(self) -> None:
        # ExitStack attempts every registered close even when an earlier close
        # fails, and preserves the readers' reverse-acquisition close order.
        stack = ExitStack()
        for reader in self._readers.values():
            stack.callback(reader.__exit__, None, None, None)
        self._readers.clear()
        self._open = False
        stack.close()
