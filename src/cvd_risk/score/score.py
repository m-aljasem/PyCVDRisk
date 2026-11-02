"""SCORE (2003): 10-year risk of fatal cardiovascular disease, low- and high-risk regions
of Europe, total-cholesterol model.

Accepts scalars or array-likes (broadcast like numpy) and returns a float64 array of risks
as fractions in [0, 1]. Cholesterol in mmol/L. A row is NaN when an input is missing or
invalid, sex/region is unrecognised, or age is outside 40-65 (the range of the charts).

Appendix A: for each cause, S0(t) = exp(-exp(alpha) (t - 20)^p); S(t) = S0(t)^exp(w);
10-year survival S(age + 10) / S(age); CVD risk = CHD risk + non-CHD risk.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk.score import coefficients as C


def _risk(alpha, p, w, age):
    # log S(age + 10) - log S(age) = -exp(alpha + w) [(age - 10)^p - (age - 20)^p] (the paper writes S0(age+10) with age - 10)
    return -np.expm1(-np.exp(alpha + w) * ((age - 10) ** p - (age - 20) ** p))


def score(age, sex, total_chol, sbp, smoker, region) -> np.ndarray:
    """SCORE 10-year fatal CVD risk. Source: coefficients.SOURCE."""
    a, sx, tc, sb, sm, rg = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_float(total_chol), as_float(sbp), as_bool(smoker),
        as_category(region, C.REGIONS))
    lo, hi = C.AGE_RANGE
    ok = (a >= lo) & (a < hi) & (sx >= 0) & (rg >= 0) & (tc > 0) & (sb > 0)
    risk = np.full(a.shape, np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        for (region, sex), base in C.BASELINE.items():
            rows = ok & (rg == C.REGIONS.index(region)) & (sx == (sex == "male"))
            total = 0.0
            for cause, (alpha, p) in base.items():
                b = C.BETA[cause]
                w = b["chol"] * (tc - 6) + b["sbp"] * (sb - 120) + b["smoker"] * sm
                total = total + _risk(alpha, p, w, a)
            risk = np.where(rows, total, risk)
    return risk
