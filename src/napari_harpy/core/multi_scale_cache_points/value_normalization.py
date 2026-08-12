from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

VALUE_NORMALIZATION_METHOD = "harpy-string-trim-unicode-white-space-case-sensitive-v1"

_UNICODE_WHITE_SPACE = "".join(
    chr(code_point)
    for start, stop in (
        (0x0009, 0x000D),
        (0x0020, 0x0020),
        (0x0085, 0x0085),
        (0x00A0, 0x00A0),
        (0x1680, 0x1680),
        (0x2000, 0x200A),
        (0x2028, 0x2029),
        (0x202F, 0x202F),
        (0x205F, 0x205F),
        (0x3000, 0x3000),
    )
    for code_point in range(start, stop + 1)
)


def _trim_utf8(values: pa.Array) -> pa.Array:
    """Trim the versioned Unicode whitespace set from UTF-8 values."""
    return pc.utf8_trim(values, characters=_UNICODE_WHITE_SPACE)


def _normalized_row_values(values: pa.Array) -> pa.Array:
    """Return normalized row-aligned string values for cache construction."""
    if pa.types.is_dictionary(values.type):
        normalized_dictionary = _trim_utf8(values.dictionary)
        return pc.take(normalized_dictionary, values.indices)
    return _trim_utf8(values)
