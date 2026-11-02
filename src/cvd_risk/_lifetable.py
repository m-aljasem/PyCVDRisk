"""Yearly life tables with a competing risk (LIFE-CVD2, DIAL2).

h, i: yearly risks of the event and of the competing death at each attained age in `ages`
(last axis); start: each person's age, shape (..., 1). Survival G is the probability of being
free of both at the start of each age, from 1 at `start`.
"""

from __future__ import annotations

import numpy as np


def survival(h, i, ages, start):
    """G at each age in `ages` and one year past the last (0 before `start`); ages grows by one."""
    ages = np.append(ages, ages[-1] + 1)
    step = 1 - np.concatenate([np.zeros(h.shape[:-1] + (1,)), h + i], axis=-1)
    step = np.where(ages > start, step, 1.0)
    return ages, np.where(ages >= start, np.cumprod(step, axis=-1), 0.0)


def lifetime_risk(h, g, ages, start, end):
    """Sum of G(t) h(t) over start <= t < end: the risk of an event before age `end`."""
    return np.sum(np.where((ages >= start) & (ages < end), g[..., :-1] * h, 0.0), axis=-1)


def median_age(g, ages, start):
    """Age at which G first falls below 0.5, linear within the year; inf if it never does.
    `g`, `ages` as returned by survival()."""
    below = (ages >= start) & (g < 0.5)
    k = np.argmax(below, axis=-1)[..., None]
    g1 = np.take_along_axis(g, k, -1)[..., 0]
    g0 = np.take_along_axis(g, np.maximum(k - 1, 0), -1)[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(below.any(axis=-1), ages[k[..., 0]] - 1 + (g0 - 0.5) / (g0 - g1), np.inf)
