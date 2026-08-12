from __future__ import annotations

import numpy as np

_UINT64_30 = np.uint64(30)
_UINT64_27 = np.uint64(27)
_UINT64_31 = np.uint64(31)
_SPLITMIX64_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)


def _splitmix64(values: np.ndarray | np.uint64) -> np.ndarray:
    """Return the vectorized SplitMix64 transform with uint64 wraparound."""
    with np.errstate(over="ignore"):
        mixed = np.asarray(values, dtype=np.uint64) + _SPLITMIX64_INCREMENT
        mixed = (mixed ^ (mixed >> _UINT64_30)) * _SPLITMIX64_MULTIPLIER_1
        mixed = (mixed ^ (mixed >> _UINT64_27)) * _SPLITMIX64_MULTIPLIER_2
    return mixed ^ (mixed >> _UINT64_31)
