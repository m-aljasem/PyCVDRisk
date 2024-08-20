"""2013 ACC/AHA Pooled Cohort Equations: 10-year risk of a first hard ASCVD event
(non-fatal MI, CHD death, fatal or non-fatal stroke).

Accepts scalars or array-likes (broadcast like numpy) and returns a float64 array of
risks as fractions in [0, 1]. Cholesterol is in mmol/L (converted to the source's mg/dL
with the same factor as PREVENT). A row is NaN when an input is missing or invalid,
sex/race is unrecognised, or age is outside 40-79.

race: "white", "african_american", or "other". The guideline (Recommendation 2) says the
equations for White individuals may be used for other groups, so "other" uses them.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk.pce.coefficients import AGE_RANGE, MODELS
from cvd_risk.prevent.prevent import MG_PER_MMOL

RACES = ("white", "african_american", "other")


def pce(age, sex, race, total_chol, hdl_chol, sbp, bp_treated, smoker, diabetes) -> np.ndarray:
    """Pooled Cohort Equations 10-year hard ASCVD risk. Source: coefficients.SOURCE."""
    a, sx, rc, tc, hdl, sbp, trt, sm, dm = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_category(race, RACES), as_float(total_chol) * MG_PER_MMOL,
        as_float(hdl_chol) * MG_PER_MMOL, as_float(sbp), as_bool(bp_treated), as_bool(smoker),
        as_bool(diabetes))
    lo, hi = AGE_RANGE
    ok = (a >= lo) & (a < hi) & (sx >= 0) & (rc >= 0) & (tc > 0) & (hdl > 0) & (sbp > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        la, lsbp = np.log(a), np.log(sbp)
        x = {"ln_age": la, "ln_age_sq": la * la, "ln_tc": np.log(tc), "ln_hdl": np.log(hdl),
             "ln_sbp_treated": lsbp * trt, "ln_sbp_untreated": lsbp * (1 - trt), "smoker": sm, "diabetes": dm}
        for k in ("tc", "hdl", "sbp_treated", "sbp_untreated", "smoker"):
            x[f"ln_age_{k}"] = la * x[k if k == "smoker" else f"ln_{k}"]
        risk = np.full(a.shape, np.nan)
        for (sex_name, race_name), c in MODELS.items():
            rows = ok & (sx == (sex_name == "male")) & (
                (rc == 1) if race_name == "african_american" else (rc != 1))
            lp = sum(c.get(k, 0.0) * v for k, v in x.items())
            risk = np.where(rows, -np.expm1(np.exp(lp - c["mean"]) * np.log(c["s0"])), risk)
    return risk
