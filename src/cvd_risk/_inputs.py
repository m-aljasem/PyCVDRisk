"""Turn user inputs (scalars, lists, numpy arrays, pandas Series) into numpy arrays.

Invalid values become NaN (numeric) or -1 (categorical) so that only the
affected row yields NaN; nothing here raises on bad row-level data.
"""

from __future__ import annotations

import numpy as np

_SEX = {"male": 1, "m": 1, "man": 1, "female": 0, "f": 0, "woman": 0}


def as_float(x) -> np.ndarray:
    """Numeric input as a float64 array; non-numeric entries become NaN."""
    try:
        return np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError):
        arr = np.asarray(x, dtype=object)
        out = np.full(arr.shape, np.nan)
        for i, v in np.ndenumerate(arr):
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                pass
        return out


def _norm(v) -> str:
    return str(v).strip().lower().replace("-", "_").replace(" ", "_")


def _map_strings(x, lookup: dict[str, int]) -> np.ndarray:
    """Map values through `lookup` after normalising; unknown -> -1. Works on unique values only."""
    uniq, inv = np.unique(np.asarray(x, dtype=object).astype(str), return_inverse=True)
    codes = np.array([lookup.get(_norm(u), -1) for u in uniq], dtype=np.int64)
    return codes[inv].reshape(np.shape(x))


def as_category(x, allowed: tuple[str, ...]) -> np.ndarray:
    """Index of each value in `allowed` (case/whitespace/-/_ insensitive); -1 if unknown."""
    return _map_strings(x, {a: i for i, a in enumerate(allowed)})


def as_sex(x) -> np.ndarray:
    """1 = male, 0 = female, -1 = unknown. Accepts 'male'/'female'/'m'/'f' (any case)."""
    return _map_strings(x, _SEX)


def as_bool(x) -> np.ndarray:
    """Binary input as float 0/1; anything other than 0/1/True/False becomes NaN."""
    v = as_float(x)
    return np.where((v == 0) | (v == 1), v, np.nan)
