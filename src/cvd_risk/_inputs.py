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


def as_category(x, allowed: tuple[str, ...]) -> np.ndarray:
    """Index of each value in `allowed` (case/whitespace/-/_ insensitive); -1 if unknown."""
    lookup = {a: i for i, a in enumerate(allowed)}
    arr = np.asarray(x, dtype=object)
    out = np.full(arr.shape, -1, dtype=np.int64)
    for i, v in np.ndenumerate(arr):
        out[i] = lookup.get(_norm(v), -1)
    return out


def as_sex(x) -> np.ndarray:
    """1 = male, 0 = female, -1 = unknown. Accepts 'male'/'female'/'m'/'f' (any case)."""
    arr = np.asarray(x, dtype=object)
    out = np.full(arr.shape, -1, dtype=np.int64)
    for i, v in np.ndenumerate(arr):
        out[i] = _SEX.get(_norm(v), -1)
    return out


def as_bool(x) -> np.ndarray:
    """Binary input as float 0/1; anything other than 0/1/True/False becomes NaN."""
    v = as_float(x)
    return np.where((v == 0) | (v == 1), v, np.nan)
