"""Compare SCORE with its official charts (Conroy 2003, Figs 1-2; fixtures/charts_transcribed.txt).

The paper prints alpha to 1 decimal and the betas to 2-3, while the charts were computed
from the unrounded function. So a chart cell is checked against the interval of risks the
printed coefficients allow: every coefficient moved by up to half a unit of its last printed
digit, in the direction that raises (or lowers) that cell's risk (risk is monotone in each).
Run from repo root: python reference/score/chart_check.py
"""

from __future__ import annotations

import copy
import importlib
from decimal import Decimal
from pathlib import Path

import numpy as np

C = importlib.import_module("cvd_risk.score.coefficients")
mod = importlib.import_module("cvd_risk.score.score")
FX = Path(__file__).parent / "fixtures" / "charts_transcribed.txt"
KNOWN_ANOMALIES = {("high", "male", 0, 65, 160, 8), ("high", "female", 0, 50, 120, 6)}


def cells():
    rows = [ln.split() for ln in FX.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    d = [(r[0], r[1], int(r[2]), int(r[3]), int(r[4]), c, int(v)) for r in rows for c, v in zip((4, 5, 6, 7, 8), r[6:])]
    return [np.array(x) for x in zip(*d)]


def _half(x):
    return float(Decimal(10) ** Decimal(repr(x)).as_tuple().exponent) / 2


def bounds(baseline=None, beta=None):
    """(lower, upper) risk in % for every chart cell under the printed-precision interval."""
    baseline, beta = baseline or C.BASELINE, beta or C.BETA
    reg, sex, sm, age, sbp, chol, _ = cells()
    out = []
    for sign in (-1, 1):
        base = {k: {cs: (a + sign * _half(a), p + sign * _half(p)) for cs, (a, p) in v.items()}
                for k, v in baseline.items()}
        risk = np.empty(len(age))
        for i in range(len(age)):
            cov = {"chol": chol[i] - 6, "sbp": sbp[i] - 120, "smoker": 1}
            b = {cs: {k: x + sign * _half(x) * np.sign(cov[k]) for k, x in v.items()} for cs, v in beta.items()}
            saved = C.BASELINE, C.BETA
            C.BASELINE, C.BETA = base, b
            try:
                risk[i] = 100 * mod.score(age[i], sex[i], chol[i], sbp[i], sm[i], reg[i])
            finally:
                C.BASELINE, C.BETA = saved
        out.append(risk)
    return out


def consistent(baseline=None, beta=None):
    *_, v = cells()
    lo, hi = bounds(baseline, beta)
    return (v >= np.floor(lo + 0.5)) & (v <= np.floor(hi + 0.5))


def main():
    reg, sex, sm, age, sbp, chol, v = cells()
    point = 100 * mod.score(age, sex, chol, sbp, sm, reg)
    ok = consistent()
    lo, hi = bounds()
    print(f"cells: {len(v)}; printed coefficients, rounded: {np.mean(np.round(point) == v):.1%} exact, "
          f"{np.mean(np.abs(np.round(point) - v) <= 1):.1%} within 1 point")
    print(f"within the printed-precision interval: {ok.sum()}/{len(v)}; interval width median "
          f"{np.median(hi - lo):.2f}, max {np.max(hi - lo):.1f} points")
    for i in np.where(~ok)[0]:
        print("outside:", reg[i], sex[i], sm[i], age[i], sbp[i], chol[i], "chart", v[i],
              f"interval {lo[i]:.2f}-{hi[i]:.2f}")
    return ok


if __name__ == "__main__":
    main()
